@echo off
for /r %%i in (Main_Course_DeepLearning_*.tex) do texify -cp %%i
