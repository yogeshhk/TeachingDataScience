@echo off
for /r %%i in (Main_Course_MLCoEP_*.tex) do texify -cp %%i
