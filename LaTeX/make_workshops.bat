@echo off
for /r %%i in (Main_Workshop*.tex) do texify -cp %%i
