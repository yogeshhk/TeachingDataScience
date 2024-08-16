@echo off
for /r %%i in (Main_DeepLearningForSa*.tex) do texify -cp %%i
