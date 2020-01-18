@echo off
for /r %%i in (Main_Sample*.tex) do texify -cp %%i
