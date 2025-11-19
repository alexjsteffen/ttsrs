# macOS Code Signing and Notarization Setup for GitHub Actions

This document explains how to set up code signing and notarization for macOS builds in GitHub Actions.

## Table of Contents

- [Code Signing Setup](#code-signing-setup)
  - [Step 1: Export Your Apple Developer Certificate](#step-1-export-your-apple-developer-certificate)
  - [Step 2: Convert Certificate to Base64](#step-2-convert-certificate-to-base64)
  - [Step 3: Get Your Code Signing Identity](#step-3-get-your-code-signing-identity)
  - [Step 4: Add Secrets to GitHub Repository](#step-4-add-secrets-to-github-repository)
  - [Step 5: Verify the Setup](#step-5-verify-the-setup)
- [Notarization Setup](#notarization-setup)
- [Security Notes](#security-notes)
- [Troubleshooting](#troubleshooting)

## Code Signing Setup

## Prerequisites

1. An Apple Developer Account with a valid **Developer ID Application** certificate
2. Access to your repository's settings on GitHub
3. macOS computer with Xcode or Xcode Command Line Tools installed (for certificate export)

## Step 1: Export Your Apple Developer Certificate

On your macOS machine:

1. Open **Keychain Access** application
2. In the left sidebar, select **login** keychain
3. Find your **Developer ID Application** certificate
4. Right-click on the certificate and select **Export "Developer ID Application: Your Name"**
5. Choose a location to save the file and set a strong password
6. Save the certificate as a `.p12` file (Personal Information Exchange)

## Step 2: Convert Certificate to Base64

Open Terminal and run the following command to convert your `.p12` file to base64:

```bash
base64 -i /path/to/your/certificate.p12 | pbcopy
```

This will copy the base64-encoded certificate to your clipboard.

## Step 3: Get Your Code Signing Identity

To find your code signing identity, run:

```bash
security find-identity -v -p codesigning
```

Look for your **Developer ID Application** certificate. The identity will look something like:

```
"Developer ID Application: Your Name (TEAM_ID)"
```

Copy the entire string including the quotes.

## Step 4: Add Secrets to GitHub Repository

1. Go to your GitHub repository
2. Navigate to **Settings** → **Secrets and variables** → **Actions**
3. Click **New repository secret** and add the following secrets:

### Required Secrets:

#### Code Signing Secrets (Required for macOS builds):

| Secret Name | Value | Description |
|-------------|-------|-------------|
| `MACOS_CERTIFICATE` | Base64-encoded certificate | The base64 string you copied in Step 2 |
| `MACOS_CERTIFICATE_PASSWORD` | Your certificate password | The password you set when exporting the .p12 file |
| `KEYCHAIN_PASSWORD` | A strong random password | Used for the temporary keychain (generate a random one) |
| `CODESIGN_IDENTITY` | Your identity string | The full identity string from Step 3, e.g., `"Developer ID Application: Your Name (TEAM_ID)"` |

#### Notarization Secrets (Optional - Choose One Method):

**Method 1: App Store Connect API Key (Recommended)**

| Secret Name | Value | Description |
|-------------|-------|-------------|
| `APPLE_API_KEY_ID` | Your Key ID | From App Store Connect (e.g., `ABC123DEF4`) |
| `APPLE_API_ISSUER_ID` | Your Issuer ID | From App Store Connect (e.g., `12345678-1234-1234-1234-123456789012`) |
| `APPLE_API_KEY` | Base64-encoded .p8 file | The base64-encoded API key file |

**Method 2: Apple ID with App-Specific Password**

| Secret Name | Value | Description |
|-------------|-------|-------------|
| `APPLE_ID` | Your Apple ID email | The email address associated with your Apple Developer account |
| `APPLE_TEAM_ID` | Your Team ID | 10-character Team ID (e.g., `A1B2C3D4E5`) |
| `APPLE_APP_SPECIFIC_PASSWORD` | App-specific password | Generated from appleid.apple.com |

**Note:** You only need to configure one of the two notarization methods. If you configure both, the API Key method will be used.

## Security Notes

## Step 5: Verify the Setup

Once you've added all the secrets:

1. Push a commit to the `master` branch
2. Check the **Actions** tab in your GitHub repository
3. The workflow should run and the macOS jobs should now include code signing steps
4. In the job logs, look for the "Code Sign Binary (macOS)" step
5. Verify that the signing was successful by checking for "codesign --verify" output

## Notarization Setup

The workflow now includes automatic notarization of the macOS binary. Notarization is an automated process where Apple scans your software for malicious content and issues a ticket that allows Gatekeeper to verify your app is safe to run.

### Setting Up Notarization

You have two options for authenticating with Apple's notarization service:

#### Option 1: App Store Connect API Key (Recommended)

This is the recommended approach as it's more secure and doesn't require manual password updates.

1. **Create an App Store Connect API Key:**
   - Go to [App Store Connect](https://appstoreconnect.apple.com)
   - Navigate to **Users and Access** → **Keys** → **App Store Connect API**
   - Click the **+** button to create a new key
   - Give it a name (e.g., "GitHub Actions Notarization")
   - Select **Developer** access level
   - Click **Generate**
   - Download the `.p8` file (you can only download it once!)

2. **Get the required information:**
   - **Key ID**: Shown in the Keys list (e.g., `ABC123DEF4`)
   - **Issuer ID**: Shown at the top of the Keys page (e.g., `12345678-1234-1234-1234-123456789012`)
   - **API Key**: The downloaded `.p8` file contents

3. **Convert the API key to base64:**
   ```bash
   base64 -i /path/to/AuthKey_ABC123DEF4.p8 | pbcopy
   ```

4. **Add the following secrets to GitHub:**
   - `APPLE_API_KEY_ID`: Your Key ID (e.g., `ABC123DEF4`)
   - `APPLE_API_ISSUER_ID`: Your Issuer ID (e.g., `12345678-1234-1234-1234-123456789012`)
   - `APPLE_API_KEY`: The base64-encoded contents of the `.p8` file

#### Option 2: Apple ID with App-Specific Password

1. **Generate an app-specific password:**
   - Go to [appleid.apple.com](https://appleid.apple.com)
   - Sign in with your Apple ID
   - Navigate to **Security** → **App-Specific Passwords**
   - Click **Generate an app-specific password**
   - Give it a name (e.g., "GitHub Actions Notarization")
   - Copy the generated password (it looks like `abcd-efgh-ijkl-mnop`)

2. **Get your Team ID:**
   - Go to your [Apple Developer Account](https://developer.apple.com/account)
   - Navigate to **Membership Details**
   - Copy your Team ID (10-character identifier like `A1B2C3D4E5`)

3. **Add the following secrets to GitHub:**
   - `APPLE_ID`: Your Apple ID email address (e.g., `developer@example.com`)
   - `APPLE_TEAM_ID`: Your Team ID (e.g., `A1B2C3D4E5`)
   - `APPLE_APP_SPECIFIC_PASSWORD`: The app-specific password you generated

### How Notarization Works in the Workflow

1. After code signing, the binary is packaged into a ZIP file
2. The ZIP is submitted to Apple's notarization service using `notarytool`
3. The workflow waits (up to 30 minutes) for Apple to complete the security scan
4. If notarization succeeds, the notarization ticket is stapled to the binary
5. The stapled binary is then uploaded as an artifact

### Skipping Notarization

If you don't configure the notarization secrets, the workflow will skip the notarization steps and only perform code signing. A warning message will be displayed in the workflow logs.

## Security Notes

- **Never commit your certificate or passwords to the repository**
- The secrets are encrypted by GitHub and only exposed to the workflow during execution
- The temporary keychain is created, used, and deleted within the same workflow run
- The certificate is only decoded into a temporary file and is cleaned up automatically

## Troubleshooting

### Error: "No identity found"

- Double-check that `CODESIGN_IDENTITY` exactly matches the output from `security find-identity`
- Ensure the identity string includes quotes if they're part of the identity name
- Verify that the temporary keychain is in the keychain search list along with the login keychain (the workflow handles this automatically)

### Error: "The specified item could not be found in the keychain"

- Verify that `MACOS_CERTIFICATE_PASSWORD` is correct
- Ensure the base64-encoded certificate in `MACOS_CERTIFICATE` is complete and not truncated

### Error: "User interaction is not allowed"

- This usually means the keychain password is incorrect or the keychain isn't properly unlocked
- Verify `KEYCHAIN_PASSWORD` is set correctly

### Certificate Expired

- Apple Developer ID certificates expire after 5 years
- You'll need to renew your certificate and update the `MACOS_CERTIFICATE` secret

### Notarization Failed

- Check the workflow logs for the detailed error message from Apple
- Common issues include:
  - Invalid API credentials or app-specific password
  - Binary not properly signed with hardened runtime
  - Binary contains unsigned or improperly signed components
  - Team ID mismatch between certificate and notarization credentials
- You can check notarization history and details at [App Store Connect](https://appstoreconnect.apple.com)

### Notarization Timeout

- The workflow waits up to 30 minutes for notarization to complete
- If Apple's service is slow, you may need to increase the timeout in the workflow
- Check Apple's [System Status](https://developer.apple.com/system-status/) page for service issues

## Additional Information

### About Developer ID Application Certificates

Developer ID Application certificates are used to sign applications that are distributed outside the Mac App Store. This allows users to run your application without macOS Gatekeeper warnings.

### About Notarization

Notarization is an automated process provided by Apple that scans your software for malicious content, security issues, and code-signing problems. When users download and run your notarized software:

1. macOS automatically checks with Apple's servers to verify the notarization ticket
2. If valid, Gatekeeper allows the app to run without warnings
3. Users get a seamless experience without security prompts

**Benefits of notarization:**
- Users can run your app without seeing "unidentified developer" warnings
- Builds trust with users that your app is safe
- Required for software distributed outside the Mac App Store on macOS 10.15 (Catalina) and later
- Helps ensure your app meets Apple's security requirements

## References

- [Apple Code Signing Guide](https://developer.apple.com/support/code-signing/)
- [Apple Notarization Guide](https://developer.apple.com/documentation/security/notarizing_macos_software_before_distribution)
- [GitHub Actions Encrypted Secrets](https://docs.github.com/en/actions/security-guides/encrypted-secrets)
- [Signing macOS Apps in GitHub Actions](https://localazy.com/blog/how-to-automatically-sign-macos-apps-using-github-actions)
