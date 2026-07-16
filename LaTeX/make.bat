@echo off
for /r %%i in (Main_Course_ML_CoEP_*.tex) do texify -cp %%i
