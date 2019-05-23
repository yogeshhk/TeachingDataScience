@echo off
for /r %%i in (Main_Workshop_Machine*.tex) do texify -cp %%i
