@echo off
for /r %%i in (Main_Course_AIML4ME*.tex) do texify -cp %%i
