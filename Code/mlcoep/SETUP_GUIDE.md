# Python Setup Guide — AIML 2026 Course

**Course**: AI-ML for Mechanical Engineers (CoEP Final Year)  
**Setup Date**: Jul 20, 2026  
**Estimated Time**: 30-45 minutes  

---

## **Prerequisites**

Before starting, ensure you have:
- A working internet connection (for downloading packages)
- ~2 GB free disk space
- Administrator access (on some systems)
- A text editor or IDE (VS Code, PyCharm, or just Jupyter will work)

---

## **Step 1: Install Anaconda or Miniconda**

### **Option A: Anaconda (Recommended for beginners)**
Anaconda includes Python, conda, and most data science packages pre-installed.

**Windows:**
1. Download from: https://www.anaconda.com/download
2. Run the installer (Anaconda3-2024.x-Windows-x86_64.exe)
3. During installation:
   - Check "Add Anaconda3 to my PATH" ✓
   - Check "Register Anaconda3 as my default Python" ✓
4. Click "Install"
5. Verify: Open Command Prompt/PowerShell and run:
   ```bash
   conda --version
   ```
   Should show: `conda 24.x.x` (or similar)

**Mac:**
1. Download from: https://www.anaconda.com/download
2. Run the installer (.pkg file)
3. Follow the installation wizard
4. Verify in Terminal:
   ```bash
   conda --version
   ```

**Linux (Ubuntu/Debian):**
```bash
wget https://repo.anaconda.com/archive/Anaconda3-2024.02-Linux-x86_64.sh
bash Anaconda3-2024.02-Linux-x86_64.sh
source ~/.bashrc
conda --version
```

---

### **Option B: Miniconda (Lightweight, recommended for experts)**
Smaller download, only includes conda and Python (you install packages as needed).

**All OS:**
1. Download from: https://docs.conda.io/en/latest/miniconda.html
2. Run the installer
3. Follow the installation wizard
4. Verify:
   ```bash
   conda --version
   ```

---

## **Step 2: Create the Course Environment**

The course uses a specific set of libraries. We've provided a `environment.yml` file with all dependencies.

### **Create Environment from YAML file:**

1. Download `environment.yml` from the course folder
2. Open Command Prompt (Windows) or Terminal (Mac/Linux)
3. Navigate to the folder where `environment.yml` is located:
   ```bash
   cd D:\Yogesh\GitHub\TeachingDataScience\Code\mlcoep
   ```
   (Replace with your actual path if different)

4. Run:
   ```bash
   conda env create -f environment.yml
   ```
   This will create an environment named `mlcoep` with all required packages.
   
   **Expected output**:
   ```
   Collecting package metadata (repodata.json): done
   Solving environment: done
   Downloading and Extracting Packages: ...
   Preparing transaction: done
   Verifying transaction: done
   Executing transaction: done
   #
   # To activate this environment, use
   #
   #     $ conda activate mlcoep
   #
   # To deactivate an active environment, use
   #
   #     $ conda deactivate
   ```

5. Activate the environment:
   ```bash
   conda activate mlcoep
   ```
   You should see `(mlcoep)` appear at the beginning of your terminal prompt.

---

## **Step 3: Verify Installation**

Run the following commands to verify all packages are installed:

```bash
python --version
```
Should show: `Python 3.9.x`

```bash
python -c "import numpy; import pandas; import sklearn; import matplotlib; import seaborn; import jupyter; print('All packages imported successfully!')"
```
Should print: `All packages imported successfully!`

If you see errors, see **Troubleshooting** section below.

---

## **Step 4: Launch Jupyter Notebook**

Jupyter is your interactive coding environment for the course.

1. Make sure the `mlcoep` environment is activated:
   ```bash
   conda activate mlcoep
   ```

2. Launch Jupyter:
   ```bash
   jupyter notebook
   ```
   
   Your default browser will open a window showing a file browser. If not, copy the URL from the terminal (looks like `http://localhost:8888/?token=...`) and paste it into your browser.

3. You should see a Jupyter interface with your home folder contents.

4. To navigate to the course materials:
   - Click through folders to reach: `D:\Yogesh\GitHub\TeachingDataScience\Code\mlcoep\notebooks\`
   - Click on a `.ipynb` notebook file to open it

5. To create a new notebook:
   - Click "New" → "Python 3" → start coding!

6. To stop Jupyter:
   - Press `Ctrl+C` in the terminal (twice if needed)

---

## **Step 5: Running Python Scripts (Optional)**

If you prefer command-line Python instead of Jupyter:

1. Activate the environment:
   ```bash
   conda activate mlcoep
   ```

2. Run a Python script:
   ```bash
   python my_script.py
   ```

3. Or use Python REPL (interactive shell):
   ```bash
   python
   >>> import numpy as np
   >>> print(np.__version__)
   >>> exit()
   ```

---

## **Environment Management**

### **Activate environment (every time you start coding):**
```bash
conda activate mlcoep
```

### **Deactivate environment (when done):**
```bash
conda deactivate
```

### **View all environments:**
```bash
conda env list
```

### **Update all packages:**
```bash
conda activate mlcoep
conda update --all
```

### **Remove environment (if you want to start fresh):**
```bash
conda env remove --name mlcoep
```

---

## **Troubleshooting**

### **Problem: "conda: command not found"**
**Solution**: Anaconda/Miniconda not installed or PATH not set correctly.
- Reinstall Anaconda and check "Add to PATH" during installation
- On Mac/Linux, run: `source ~/.bashrc` or `source ~/.zshrc` after installation

---

### **Problem: "ModuleNotFoundError: No module named 'numpy'" (or pandas, sklearn, etc.)**
**Solution**: Package not installed in the environment.
- Verify you're in the `mlcoep` environment: `conda activate mlcoep`
- Check that `(mlcoep)` appears in your prompt
- Install missing package: `conda install numpy` (or the package name)

---

### **Problem: "Port 8888 is already in use" (Jupyter error)**
**Solution**: Another Jupyter instance is running.
- Option 1: Stop the other Jupyter (find the terminal window and press `Ctrl+C`)
- Option 2: Use a different port: `jupyter notebook --port 8889`

---

### **Problem: Jupyter won't open in browser**
**Solution**: Copy the URL manually.
- After running `jupyter notebook`, you'll see output like:
  ```
  http://localhost:8888/?token=abc123def456...
  ```
- Copy this URL and paste it into your browser (Chrome, Firefox, Safari, Edge)

---

### **Problem: "PermissionError" or "Access Denied" during installation**
**Solution**: Need administrator privileges.
- **Windows**: Open Command Prompt as Administrator (right-click → "Run as administrator")
- **Mac/Linux**: Use `sudo` (if needed): `sudo conda install package_name`

---

### **Problem: Environment creation fails ("Solving environment: |")**
**Solution**: Conda is stuck solving dependencies.
- Press `Ctrl+C` to cancel
- Try with `--no-deps` flag (less safe): `conda env create -f environment.yml --no-deps`
- Or update conda: `conda update -n base -c defaults conda`

---

## **Need Help?**

### **Course Support:**
- **Email**: (To be provided by instructor)
- **Office Hours**: (To be provided by instructor)
- **Course Group**: Google Group (mlcoep2026@googlegroups.com)

### **Online Resources:**
- **Conda Docs**: https://conda.io/projects/conda/en/latest/user-guide/
- **Jupyter Docs**: https://jupyter.org/
- **NumPy Tutorial**: https://numpy.org/doc/
- **Pandas Tutorial**: https://pandas.pydata.org/docs/

### **Quick Python Cheat Sheets:**
- Pandas: https://pandas.pydata.org/docs/getting_started/cheatsheet.html
- NumPy: https://numpy.org/doc/stable/user/basics.broadcasting.html

---

## **Common Commands Quick Reference**

```bash
# Activate course environment
conda activate mlcoep

# Launch Jupyter
jupyter notebook

# Run Python script
python script.py

# Install a new package
conda install package_name

# Update a package
conda update package_name

# List installed packages
conda list

# Deactivate environment
conda deactivate
```

---

## **Next Steps**

Once your environment is set up:
1. Launch Jupyter: `jupyter notebook`
2. Navigate to: `mlcoep/notebooks/practice/`
3. Open the first practice notebook: `01_Python_Fundamentals.ipynb`
4. Start coding!

---

**Happy coding! 🚀**

**Last Updated**: Jul 16, 2026  
**Tested on**: Python 3.9.x, Windows 11 / Mac / Linux
