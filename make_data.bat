@echo off
for /r %%i in (Main_Workshop_Data*.tex) do texify -cp %%i
