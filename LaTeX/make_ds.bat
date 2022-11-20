@echo off
for /r %%i in (Main_*_DataScience*.tex) do texify -cp %%i
