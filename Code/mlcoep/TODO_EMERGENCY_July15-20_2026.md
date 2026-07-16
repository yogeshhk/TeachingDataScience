# EMERGENCY TO-DO: Course Launch Week (July 15-20, 2026)

**Course Starts**: Friday, July 20, 2026  
**Prep Window**: Mon Jul 15 – Thu Jul 19 (5 working days)  
**Status**: CRITICAL PATH ONLY

---

## **CRITICAL: MUST BE DONE BY JULY 20**

### **1. Compile LaTeX Slides to PDF** ⚠️ URGENT
**Deadline**: Tue Jul 16 (2 days)  
**Effort**: 1-2 hours  
**Owner**: Self

**Tasks**:
- [ ] Open LaTeX folder: `D:\Yogesh\GitHub\TeachingDataScience\LaTeX\`
- [ ] Compile Presentation PDF:
  ```bash
  cd LaTeX
  texify -cp Main_Course_ML_CoEP_Presentation.tex
  ```
- [ ] Compile CheatSheet PDF:
  ```bash
  texify -cp Main_Course_ML_CoEP_CheatSheet.tex
  ```
- [ ] Verify both PDFs generated correctly (check page count, no errors)
- [ ] Copy both PDFs to `Code/mlcoep/slides/` folder (create if needed)
- [ ] Test opening in Adobe Reader / PDF viewer

**Output Files Expected**:
- `Main_Course_ML_CoEP_Presentation.pdf` (24 sessions)
- `Main_Course_ML_CoEP_CheatSheet.pdf` (landscape, 2-3 columns)

**Blocking**: Everything else depends on having slides ready

---

### **2. Create Python Setup Guide** ⚠️ URGENT
**Deadline**: Tue Jul 16 (2 days)  
**Effort**: 1 hour  
**Owner**: Self

**Tasks**:
- [ ] Create file: `Code/mlcoep/SETUP_GUIDE.md`
- [ ] Include:
  - [ ] Python 3.8+ installation (Windows/Mac/Linux)
  - [ ] Anaconda/Miniconda setup
  - [ ] Create conda environment from yml file (template below)
  - [ ] Install required libraries: numpy, pandas, scikit-learn, matplotlib, jupyter
  - [ ] Verify installation (test `import` statements)
  - [ ] Jupyter Notebook launch instructions
  - [ ] Troubleshooting common errors (import errors, PATH issues)
  - [ ] Contact info for help

**Conda Environment File** (`environment.yml`):
```yaml
name: mlcoep
channels:
  - conda-forge
dependencies:
  - python=3.9
  - numpy
  - pandas
  - scikit-learn
  - matplotlib
  - seaborn
  - jupyter
  - ipython
  - pip
  - pip:
    - kaggle
```

**Output**: 
- `Code/mlcoep/environment.yml` (ready to share)
- `Code/mlcoep/SETUP_GUIDE.md` (instructions)

**Blocking**: Students need this on Day 1

---

### **3. Create Assignment A1** ⚠️ URGENT
**Deadline**: Wed Jul 17 (3 days)  
**Effort**: 2 hours  
**Owner**: Self

**Tasks**:
- [ ] Create folder: `Code/mlcoep/assignments/A1_Python_Basics/`
- [ ] Create files:
  - [ ] `A1_Problem_Statement.md` — 3 coding problems
    1. Fibonacci Series (10 lines)
    2. Quadratic Equation Solver (with error handling)
    3. Data type classification (given data → classify by type)
  - [ ] `A1_Solution.py` — working solution code
  - [ ] `A1_Rubric.md` — grading rubric (40% correctness, 20% clarity, 20% efficiency, 20% documentation)
  - [ ] `A1_README.md` — submission format, due date (Jul 27), examples

**Output**: Ready to hand out Day 1 (Jul 20)

**Blocking**: Students need assignment to start coding immediately

---

### **4. Prepare First Class Materials** ⚠️ URGENT
**Deadline**: Thu Jul 18 (4 days)  
**Effort**: 1.5 hours  
**Owner**: Self

**Tasks**:
- [ ] Print/prepare:
  - [ ] Attendance sheet (24 sessions + blank rows for names)
  - [ ] Course syllabus (README_AIML4ME_2026.md) — printed or digital copy
  - [ ] Python setup guide (SETUP_GUIDE.md)
  - [ ] Assignment A1 (problem statement)
  - [ ] Course schedule (sessions 1-24, dates, topics)

- [ ] Digital copies ready:
  - [ ] Email attendance sheet template to self
  - [ ] Organize all PDFs in folder: `Code/mlcoep/first_class/`

- [ ] Classroom logistics:
  - [ ] Confirm room 301, Bajaj Building
  - [ ] Test projector for PDF display
  - [ ] Have USB backup of slides + materials

**Output**: All materials in `Code/mlcoep/first_class/` + printed copies

---

### **5. Create Datasets Folder Structure** ⚠️ MEDIUM PRIORITY
**Deadline**: Thu Jul 18 (4 days)  
**Effort**: 1 hour  
**Owner**: Self

**Tasks**:
- [ ] Create folder: `Code/mlcoep/datasets/`
- [ ] Create subfolders:
  - [ ] `manufacturing_equipment/` — equipment logs (download from Kaggle by Jul 25)
  - [ ] `quality_metrics/` — quality data (download by Aug 5)
  - [ ] `predictive_maintenance/` — bearing/CMAPSS (download by Aug 15)
  - [ ] `capstone/` — steel plates or equipment lifespan (download by Sep 15)

- [ ] Create `README.md` in each folder:
  - [ ] Dataset source (URL)
  - [ ] Download instructions
  - [ ] Data dictionary (columns, types, ranges)
  - [ ] Size (rows, columns, file format)
  - [ ] License / usage rights

**Output**: Folder structure ready, download links documented

**Note**: Actual datasets can be downloaded during course (not blocking start)

---

## **HIGH PRIORITY: FIRST WEEK OF COURSE (Jul 20-27)**

- [ ] **T-SESSIONS-1-4**: Have Sessions 1-4 slides polished + printed (first 2 weeks)
- [ ] **T-A1-SUBMIT**: Collect A1 submissions (due Jul 27)
- [ ] **T-GRADE-A1**: Grade A1 by Aug 1 (quick turnaround, motivates students)
- [ ] **T-ATTENDANCE**: Take attendance daily (build habit)

---

## **MEDIUM PRIORITY: PARALLEL DURING COURSE**

**These can be done during course delivery (don't block start)**:

- [ ] **T-DATASETS**: Download actual datasets (Aug-Sep)
- [ ] **T-ASSIGNMENTS-A2-A6**: Finalize A2-A6 one week before each due date
- [ ] **T-EXAMS**: Design midterm (complete by late Jul, give in late Aug)
- [ ] **T-RUBRICS**: Finalize assignment/project/exam rubrics by early Aug
- [ ] **T-STUDENT-HANDBOOK**: Create by early Aug
- [ ] **T-TRACKERS**: Set up grade tracker by early Aug

---

## **DEFERRED: Can wait (not blocking July 20 start)**

- Midterm exam finalization (have outline, finalize mid-Jul)
- EndSem exam (finalize by Oct)
- Capstone project options (finalize by Sep)
- Industry guest lectures (arrange by Aug)
- Question bank expansion (nice-to-have, not critical)

---

## **DAILY CHECKLIST: Jul 15-20**

### **Monday, Jul 15**
- [ ] List all files needed for first class
- [ ] Locate LaTeX folder and test compilation
- [ ] Create `Code/mlcoep/` folder structure

### **Tuesday, Jul 16** ← CRITICAL DAY
- [ ] ✅ Compile LaTeX to PDF (presentation + cheatsheet)
- [ ] ✅ Create Python setup guide + environment.yml
- [ ] Verify PDF files exist and are readable

### **Wednesday, Jul 17** ← CRITICAL DAY
- [ ] ✅ Create Assignment A1 (problem + solution + rubric)
- [ ] Create datasets folder structure
- [ ] Double-check all PDFs print correctly

### **Thursday, Jul 18**
- [ ] ✅ Print all first-class materials
- [ ] ✅ Organize digital copies in folders
- [ ] Test projector setup (if possible)
- [ ] Prepare attendance sheet

### **Friday, Jul 20** ← COURSE STARTS
- [ ] Arrive early to classroom
- [ ] Test projector + slides display
- [ ] Have printed materials ready
- [ ] Greet students, hand out syllabus + setup guide + A1
- [ ] Take attendance
- [ ] Present Session 1 (AI Overview)

---

## **SUCCESS CRITERIA: Jul 20**

✅ **By 10:30 AM Jul 20, I will have**:
- [ ] Working PDF slides (Sessions 1-24)
- [ ] Python setup guide printed + digital
- [ ] Assignment A1 ready to distribute
- [ ] Attendance sheet
- [ ] Course syllabus copies
- [ ] Projector tested & working
- [ ] All materials organized in `Code/mlcoep/`

✅ **Students will leave with**:
- [ ] Course syllabus
- [ ] Python setup instructions
- [ ] Assignment A1 (due Jul 27)
- [ ] Course schedule
- [ ] Contact info for questions

---

## **EMERGENCY CONTACTS / HELP**

**If LaTeX compilation fails**:
- Check MikTeX installation: `miktex --version`
- Run `miktex update` to refresh packages
- Try `pdflatex` directly instead of `texify`
- Fallback: Use existing 2025 slides as placeholder (less ideal)

**If datasets unavailable**:
- Delay until needed (not blocking start)
- Use synthetic/sample data in first assignments
- Create fallback toy datasets

**If setup guide unclear**:
- Test personally (run through setup steps yourself)
- Get feedback from one student on Jul 20
- Refine by Jul 25

---

## **TIME BUDGET**

| Task | Hours | Start | Finish |
|------|-------|-------|--------|
| LaTeX compile | 1-2h | Tue morning | Tue afternoon |
| Setup guide | 1h | Tue afternoon | Tue evening |
| Assignment A1 | 2h | Wed morning | Wed afternoon |
| First class prep | 1.5h | Thu morning | Thu afternoon |
| Dataset structure | 1h | Thu afternoon | Thu evening |
| **TOTAL** | **~6.5h** | **Jul 15** | **Jul 18** |

**Realistic pace**: 1-2 hours/day over 5 days = ACHIEVABLE ✅

---

**STATUS**: Ready to execute  
**CONFIDENCE**: HIGH (most materials already exist, just need compilation + assembly)  
**RISK**: LOW (only risk is LaTeX compilation issues, have fallback)

**LET'S GO! 🚀**
