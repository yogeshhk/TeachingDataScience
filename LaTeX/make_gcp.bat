@echo off
for /r %%i in (Main_Seminar_GCP*.tex) do texify -cp %%i
