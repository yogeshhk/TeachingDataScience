@echo off
for /r %%i in (Main_Seminar_Graph_GeometricDeepLearning_*.tex) do texify -cp %%i
