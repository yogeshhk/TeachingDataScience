@echo off
for /r %%i in (Main_*DIET*.tex) do texify -cp %%i
