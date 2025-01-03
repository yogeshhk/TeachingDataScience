@echo off
for /r %%i in (Main_Course_PyML4ME*.tex) do texify -cp %%i
