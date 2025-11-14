@echo off
for /r %%i in (Main_Seminar_NLP_ML_*.tex) do texify -cp %%i
