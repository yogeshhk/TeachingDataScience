@echo off
for /r %%i in (Main_*Python*.tex) do texify -cp %%i
