# Pushing Wiki to GitHub

The wiki content is ready in `/Users/josueortegacaro/repos/prismt.wiki/`. 

## Option 1: Push via Command Line (Recommended)

If the wiki repository exists, run:

```bash
cd /Users/josueortegacaro/repos/prismt.wiki
git push -u origin main
```

## Option 2: Create First Page via GitHub UI

If the push fails, GitHub may need the wiki repository to be initialized first:

1. Go to: https://github.com/josueortc/prismt/wiki
2. Click "Create the first page" or "Edit" if a page exists
3. Add any content (even just "Test") and save
4. This will create the wiki repository
5. Then clone and push:

```bash
# Clone the newly created wiki repo
cd /Users/josueortegacaro/repos
rm -rf prismt.wiki  # Remove local copy
git clone https://github.com/josueortc/prismt.wiki.git

# Copy wiki files
cp /Users/josueortegacaro/repos/prismt.wiki/*.md /Users/josueortegacaro/repos/prismt.wiki/

# Commit and push
cd /Users/josueortegacaro/repos/prismt.wiki
git add *.md
git commit -m "Add comprehensive wiki documentation"
git push origin main
```

## Option 3: Manual Upload via GitHub UI

You can also manually create each page:

1. Go to: https://github.com/josueortc/prismt/wiki
2. Click "New Page" for each file:
   - Home.md → Create "Home" page
   - Data-Standardization-Overview.md → Create "Data-Standardization-Overview" page
   - Preparing-Widefield-Data.md → Create "Preparing-Widefield-Data" page
   - Preparing-CDKL5-Data.md → Create "Preparing-CDKL5-Data" page
   - Standardized-Data-Format.md → Create "Standardized-Data-Format" page
   - Validation-and-Troubleshooting.md → Create "Validation-and-Troubleshooting" page
   - Complete-Workflow-Examples.md → Create "Complete-Workflow-Examples" page

3. Copy the content from each `.md` file in `/Users/josueortegacaro/repos/prismt.wiki/`

## Verify

After pushing, verify the wiki is accessible at:
https://github.com/josueortc/prismt/wiki
