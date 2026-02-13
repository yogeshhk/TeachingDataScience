@echo off
for /r %%i in (Main_Seminar_ML_Presentation*.tex) do texify -cp %%i
