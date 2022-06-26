@echo off
for /r %%i in (Main_CoEP*.tex) do texify -cp %%i
