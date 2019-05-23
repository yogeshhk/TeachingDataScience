@echo off
for /r %%i in (Main_Workshop_Maths*.tex) do texify -cp %%i
