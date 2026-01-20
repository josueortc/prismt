# Push PRISMT Repository to GitHub

## Option 1: Create Repository Manually (Recommended)

1. **Go to GitHub**: https://github.com/new

2. **Repository Settings**:
   - Repository name: `prismt`
   - Description: `Unified Pipeline for Widefield and CDKL5 Data`
   - Visibility: **Select "Private"** ✅
   - **DO NOT** check "Initialize with README" (we already have one)
   - **DO NOT** add .gitignore or license

3. **Click "Create repository"**

4. **Push the code**:
   ```bash
   cd /Users/josueortegacaro/repos/prismt
   git push -u origin main
   ```

## Option 2: Create Repository via GitHub API

If you have a GitHub Personal Access Token:

1. **Get a GitHub Token**:
   - Go to: https://github.com/settings/tokens
   - Click "Generate new token (classic)"
   - Select scope: `repo` (full control of private repositories)
   - Generate and copy the token

2. **Set the token**:
   ```bash
   export GITHUB_TOKEN=your_token_here
   ```

3. **Run the script**:
   ```bash
   cd /Users/josueortegacaro/repos/prismt
   ./create_github_repo.sh
   ```

## Verify

After pushing, verify the repository:
```bash
git remote -v
```

You should see:
```
origin  https://github.com/josueortc/prismt.git (fetch)
origin  https://github.com/josueortc/prismt.git (push)
```

Visit: https://github.com/josueortc/prismt to see your repository.
