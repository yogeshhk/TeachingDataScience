@echo off
for /r %%i in (Main_Course_Machine*.tex) do texify -cp %%i
