# macOS Code Signing Setup for GitHub Actions

This document explains how to set up code signing for macOS builds in GitHub Actions.

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

| Secret Name | Value | Description |
|-------------|-------|-------------|
| `MACOS_CERTIFICATE` | Base64-encoded certificate | The base64 string you copied in Step 2 |
| `MACOS_CERTIFICATE_PASSWORD` | Your certificate password | The password you set when exporting the .p12 file |
| `KEYCHAIN_PASSWORD` | A strong random password | Used for the temporary keychain (generate a random one) |
| `CODESIGN_IDENTITY` | Your identity string | The full identity string from Step 3, e.g., `"Developer ID Application: Your Name (TEAM_ID)"` |

## Step 5: Verify the Setup

Once you've added all the secrets:

1. Push a commit to the `master` branch
2. Check the **Actions** tab in your GitHub repository
3. The workflow should run and the macOS jobs should now include code signing steps
4. In the job logs, look for the "Code Sign Binary (macOS)" step
5. Verify that the signing was successful by checking for "codesign --verify" output

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

## Additional Information

### About Developer ID Application Certificates

Developer ID Application certificates are used to sign applications that are distributed outside the Mac App Store. This allows users to run your application without macOS Gatekeeper warnings.

### Notarization (Optional)

For full macOS compatibility, you may also want to notarize your application with Apple. This requires:
- Uploading the signed binary to Apple's notarization service
- Waiting for Apple's approval
- Stapling the notarization ticket to your binary

This is not currently implemented in the workflow but can be added as a future enhancement.

## References

- [Apple Code Signing Guide](https://developer.apple.com/support/code-signing/)
- [GitHub Actions Encrypted Secrets](https://docs.github.com/en/actions/security-guides/encrypted-secrets)
- [Signing macOS Apps in GitHub Actions](https://localazy.com/blog/how-to-automatically-sign-macos-apps-using-github-actions)
