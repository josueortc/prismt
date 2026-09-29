# Wiki Content

This folder contains wiki-ready content for the [PRISMT GitHub Wiki](https://github.com/josueortc/prismt/wiki).

## Adding the MATLAB GUI Tutorial to the Wiki

1. **Clone the wiki repo** (if not already):
   ```bash
   git clone https://github.com/josueortc/prismt.wiki.git
   cd prismt.wiki
   ```

2. **Create the new page**:
   - Copy `MATLAB-GUI-Tutorial.md` to `MATLAB-GUI-Tutorial.md` in the wiki repo
   - Or create a new page in the wiki web interface and paste the content

3. **Add screenshots**:
   - Generate screenshots: In MATLAB, run `capture_gui_screenshots` (see [MATLAB_GUI_Tutorial.md](../MATLAB_GUI_Tutorial.md))
   - Create an `images/gui/` folder in the wiki repo
   - Upload the PNG files (1_initial.png, 2_path_entered.png, etc.)
   - Update image paths in the wiki page if needed

4. **Link from the wiki Home**:
   - Edit `Home.md` and add to Quick Navigation:
   ```markdown
   - [MATLAB GUI Tutorial](MATLAB-GUI-Tutorial) - Frame-by-frame guide
   ```

5. **Commit and push** (if using wiki repo):
   ```bash
   git add .
   git commit -m "Add MATLAB GUI Tutorial"
   git push
   ```

## Adding the Hyperparameter Guide to the Wiki

1. **Clone the wiki repo** (if not already): same as above.

2. **Create the new page**:
   - Copy `Hyperparameter-Guide.md` to `Hyperparameter-Guide.md` in the wiki repo
   - Or create a new page in the wiki web interface and paste the content

3. **Link from the wiki Home**:
   - Edit `Home.md` and add to Quick Navigation:
   ```markdown
   - [Hyperparameter Guide](Hyperparameter-Guide) - What each hyperparameter does and how to tune it
   ```

4. **Commit and push** (if using wiki repo):
   ```bash
   git add .
   git commit -m "Add Hyperparameter Guide"
   git push
   ```
