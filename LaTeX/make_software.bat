@echo off
for /r %%i in (Main_Workshop_Software*.tex) do texify -cp %%i
