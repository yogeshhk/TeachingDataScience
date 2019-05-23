@echo off
for /r %%i in (Main_Workshop_Deep*.tex) do texify -cp %%i
