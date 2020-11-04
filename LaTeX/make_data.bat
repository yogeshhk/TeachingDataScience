@echo off
for /r %%i in (Main_*_Data*.tex) do texify -cp %%i
