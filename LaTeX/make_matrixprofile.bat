@echo off
for /r %%i in (Main_*.tex) do texify -cp %%i
