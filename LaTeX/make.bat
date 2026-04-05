@echo off
for /r %%i in (Main_Seminar_*Educators*.tex) do texify -cp %%i
