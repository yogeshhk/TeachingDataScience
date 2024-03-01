@echo off
for /r %%i in (Main_*_Blockchain*.tex) do texify -cp %%i
