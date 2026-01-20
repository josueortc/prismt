# Setting Up GitHub Repository

The PRISMT repository has been initialized locally. To push it to GitHub, follow these steps:

## Option 1: Create Repository via GitHub Website

1. Go to https://github.com/new
2. Repository name: `prismt`
3. Description: "Unified Pipeline for Widefield and CDKL5 Data"
4. Choose Public or Private
5. **DO NOT** initialize with README, .gitignore, or license (we already have these)
6. Click "Create repository"

## Option 2: Create Repository via GitHub CLI

If you have GitHub CLI installed:

```bash
gh repo create prismt --public --description "Unified Pipeline for Widefield and CDKL5 Data"
```

## Push to GitHub

After creating the repository on GitHub, run:

```bash
cd /Users/josueortegacaro/repos/prismt
git remote add origin https://github.com/josueortc/prismt.git
git branch -M main
git push -u origin main
```

Or if you prefer SSH:

```bash
git remote add origin git@github.com:josueortc/prismt.git
git branch -M main
git push -u origin main
```

## Verify

After pushing, verify the repository is accessible:

```bash
git remote -v
```

You should see:
```
origin  https://github.com/josueortc/prismt.git (fetch)
origin  https://github.com/josueortc/prismt.git (push)
```
