@echo off
for /r %%i in (Main_Seminar_AI-ML_*.tex) do texify -cp %%i
