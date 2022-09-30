@echo off
for /r %%i in (Main_*Graph*.tex) do texify -cp %%i
