@echo off
for /r %%i in (Main_Seminar_Arti*.tex) do texify -cp %%i
