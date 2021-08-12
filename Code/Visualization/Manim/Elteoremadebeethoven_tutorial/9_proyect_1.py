from big_ol_pile_of_manim_imports import *
from tutorial.formulas_txt.formulas import *
#Inicia parametros -----------------------------------------

tiempo=1.5
tiempo_e=1

class DespejeEcuacion2doGrado(Scene):
	CONFIG ={
	"camera_config" : {"background_color":"#272727"}
	}
	def construct(self):
		self.importar_formulas()
		self.imprime_formula()
		self.paso_1(tiempo_e)
		self.paso_2(tiempo_e)
		self.paso_3(tiempo_e)
		self.paso_4(tiempo_e)
		self.paso_5(tiempo_e)
		self.paso_6(tiempo_e)
		self.paso_7(tiempo_e)
		self.paso_8(tiempo_e)
		self.paso_9(tiempo_e)
		self.paso_10(tiempo_e)
		self.paso_11(tiempo_e)
		self.paso_12(tiempo_e)
		self.paso_13(tiempo_e)
		self.paso_14(tiempo_e)
		#
		c1=SurroundingRectangle(self.formulas[14],buff=0.2)
		c2=SurroundingRectangle(self.formulas[14],buff=0.2)
		c2.rotate(PI)
		self.play(ShowCreationThenDestruction(c1),ShowCreationThenDestruction(c2))
		self.wait(2)
		#'''

	def importar_formulas(self):
		#self.formulas=[]
		#for i in range(len(formulas)):
		#	self.formulas.append(TexMobject(*formulas[i]).scale(1.7))
		self.formulas=formulas


	def imprime_formula(self):
		self.play(Write(self.formulas[0]))

	def paso_1(self,tiempo_e):
		#Parámetros
		paso=1
		cambios = [[
						(	0,	1,	3,	4,	5,	6,	7,	8,	9	),
						(	0,	1,	3,	4,	5,	6,	8,	9,	7	)
		]]
		write = []
		fade = [10]
		arco=-PI/2

		#Inicia escena------------------------
		self.add(self.formulas[paso-1])
		for pre_ind,post_ind in cambios:
			self.play(*[
				ReplacementTransform(
					self.formulas[paso-1][i],self.formulas[paso][j],
					path_arc=arco
					)
				for i,j in zip(pre_ind,post_ind)
				],
				*[FadeOut(self.formulas[paso-1][f])for f in fade],
				*[Write(self.formulas[paso][w])for w in write],
				run_time=tiempo
			)
			self.wait(tiempo_e)

	def paso_2(self,tiempo_e):
		#Parámetros
		paso=2
		cambios = [[
						(	0,		1,	3,	4,	5,	6,	7,	8,	9	),
						(	7,		0,	2,	3,	5,	9,	10,	11,	13	)
		]]
		write = [6,14]
		fade = []

		arco=0

		copias=[0]
		fin=[15]
		formula_copia=[] #No modificar
		for c in copias: #No modificar
			formula_copia.append(self.formulas[paso-1][c].copy()) #No modificar
		#Inicia escena------------------------
		self.add(self.formulas[paso-1])
		for pre_ind,post_ind in cambios:
			self.play(*[
				ReplacementTransform(
					self.formulas[paso-1][i],self.formulas[paso][j],
					path_arc=arco
					)
				for i,j in zip(pre_ind,post_ind)
				],
				*[FadeOut(self.formulas[paso-1][f])for f in fade],
				*[Write(self.formulas[paso][w])for w in write],
				#'''
				*[ReplacementTransform(formula_copia[j],self.formulas[paso][f])
				for j,f in zip(range(len(fin)),fin)
				],
				#'''
				run_time=tiempo
			)
			self.wait(tiempo_e)

	def paso_3(self,tiempo_e):
		#Parámetros
		paso=3
		cambios = [[
			(	0,	2,	3,	5,	6,	7,	9,	10,	11,	13,	14,	15	),
			(	0,	2,	3,	5,	6,	7,	9,	21,	22,	24,	25,	26	)
		]]
		write = [10,	11,	13,	14,	15,	16,	18,	20,	28,	29,	31,	32,	33,	34,	36,	38]
		fade = []

		arco=0

		copias=[]
		fin=[]
		formula_copia=[] #No modificar
		for c in copias: #No modificar
			formula_copia.append(self.formulas[paso-1][c].copy()) #No modificar
		#Inicia escena------------------------
		self.add(self.formulas[paso-1])
		for pre_ind,post_ind in cambios:
			self.play(*[
				ReplacementTransform(
					self.formulas[paso-1][i],self.formulas[paso][j],
					path_arc=arco
					)
				for i,j in zip(pre_ind,post_ind)
				],
				*[FadeOut(self.formulas[paso-1][f])for f in fade],
				#*[Write(self.formulas[paso][w])for w in write],
				#'''
				*[ReplacementTransform(formula_copia[j],self.formulas[paso][f])
				for j,f in zip(range(len(fin)),fin)
				],
				#'''
				run_time=tiempo
			)
			self.wait()
		self.play(*[Write(self.formulas[paso][w])for w in write])
		#self.play(*[FadeOut(self.formulas[paso-1][f])for f in fade])
		self.wait(tiempo_e)

	def paso_4(self,tiempo_e):
		#Parámetros
		paso=4
		cambios = [[
				(	0,	2,	10,	11,	13,	14,	15,	16,	18,	20,	21,	22,	24,	25,	26,	28,	29,	31,	32,	33,	34,	36,	38	),
				(	1,	11,	2,	0,	4,	5,	6,	7,	9,	11,	12,	13,	15,	16,	17,	19,	20,	22,	23,	24,	25,	27,	29	)
		]]
		write = []
		fade = [3,	5,	6,	7,	9]

		arco=0

		copias=[]
		fin=[]
		formula_copia=[] #No modificar
		for c in copias: #No modificar
			formula_copia.append(self.formulas[paso-1][c].copy()) #No modificar
		#Inicia escena------------------------
		self.add(self.formulas[paso-1])
		for pre_ind,post_ind in cambios:
			self.play(*[
				ReplacementTransform(
					self.formulas[paso-1][i],self.formulas[paso][j],
					path_arc=arco
					)
				for i,j in zip(pre_ind,post_ind)
				],
				*[FadeOut(self.formulas[paso-1][f])for f in fade],
				*[Write(self.formulas[paso][w])for w in write],
				#------------Copias
				*[ReplacementTransform(formula_copia[j],self.formulas[paso][f])
				for j,f in zip(range(len(fin)),fin)
				],
				#
				run_time=tiempo
			)
			self.wait(tiempo_e)

	def paso_5(self,tiempo_e):
		#Parámetros
		paso=5
		cambios = [[
		(	0,	1,	2,	4,	5,	6,	7,	9,	11,	12,	13,	15,	16,	17,	19,	22,	23,	24,	25,	29),
		(	0,	1,	2,	4,	5,	6,	7,	9,	11,	12,	13,	15,	16,	17,	19,	21,	24,	25,	26,	23)
		]]
		write = []
		fade = [20,27]

		arco=0

		copias=[29]
		copia_fin=[28]
		formula_copia=[] #No modificar
		for c in copias: #No modificar
			formula_copia.append(self.formulas[paso-1][c].copy()) #No modificar
		#Inicia escena------------------------
		self.add(self.formulas[paso-1])
		for pre_ind,post_ind in cambios:
			self.play(*[
				ReplacementTransform(
					self.formulas[paso-1][i],self.formulas[paso][j],
					path_arc=arco
					)
				for i,j in zip(pre_ind,post_ind)
				],
				*[FadeOut(self.formulas[paso-1][f])for f in fade],
				#*[Write(self.formulas[paso][w])for w in write],
				*[ReplacementTransform(formula_copia[j],self.formulas[paso][f])
				for j,f in zip(range(len(copia_fin)),copia_fin)
				],
				run_time=tiempo
			)
			self.wait(tiempo_e)

	def paso_6(self,tiempo_e):
		#Parámetros
		paso=6
		cambios = [[
			(	0,	1,	2,	4,	5,	6,	7,	9,	11,	12,	13,	15,	16,	17,	21,	23,	24,	25,	26,	28	),
			(	0,	1,	2,	4,	5,	6,	7,	9,	11,	12,	23,	25,	26,	27,	14,	16,	17,	18,	19,	21	)
		]]
		write = []
		fade = [19]

		arco=0

		copias=[]
		copia_fin=[]
		formula_copia=[] #No modificar
		for c in copias: #No modificar
			formula_copia.append(self.formulas[paso-1][c].copy()) #No modificar
		#Inicia escena------------------------
		self.add(self.formulas[paso-1])
		for pre_ind,post_ind in cambios:
			self.play(*[
				ReplacementTransform(
					self.formulas[paso-1][i],self.formulas[paso][j],
					path_arc=arco
					)
				for i,j in zip(pre_ind,post_ind)
				],
				*[FadeOut(self.formulas[paso-1][f])for f in fade],
				#*[Write(self.formulas[paso][w])for w in write],
				*[ReplacementTransform(formula_copia[j],self.formulas[paso][f])
				for j,f in zip(range(len(copia_fin)),copia_fin)
				],
				run_time=tiempo
			)
			self.wait(tiempo_e)

	def paso_7(self,tiempo_e):
		#Parámetros
		paso=7
		cambios = [[
			(	0,	1,	2,		4,	5,	6,	7,		9,		11,	12,		14,		16,	17,	18,	19,		21,		23,		25,	26,	27	),
			(	0,	1,	2,		4,	5,	6,	7,		9,		11,	12,		14,		16,	17,	18,	19,		21,		23,		26,	27,	29	)
		]]
		write = [25,28]
		fade = []

		arco=0

		copias=[]
		copia_fin=[]
		formula_copia=[] #No modificar
		for c in copias: #No modificar
			formula_copia.append(self.formulas[paso-1][c].copy()) #No modificar
		#Inicia escena------------------------
		self.add(self.formulas[paso-1])
		for pre_ind,post_ind in cambios:
			self.play(*[
				ReplacementTransform(
					self.formulas[paso-1][i],self.formulas[paso][j],
					path_arc=arco
					)
				for i,j in zip(pre_ind,post_ind)
				],
				*[FadeOut(self.formulas[paso-1][f])for f in fade],
				#*[Write(self.formulas[paso][w])for w in write],
				*[ReplacementTransform(formula_copia[j],self.formulas[paso][f])
				for j,f in zip(range(len(copia_fin)),copia_fin)
				],
				run_time=tiempo
			)
			self.wait(tiempo_e)
		self.play(*[Write(self.formulas[paso][w])for w in write])
		#self.play(*[FadeOut(self.formulas[paso-1][f])for f in fade])
		#self.wait()

	def paso_8(self,tiempo_e):
		#Parámetros
		paso=8
		cambios = [[
			(	0,	1,	2,		4,	5,	6,	7,		9,		11,	12,		14,		16,	17,	18,	19,		21,		23,		25,	26,	27,	28,	29,	),
			(	0,	1,	2,		4,	5,	6,	7,		9,		11,	12,		14,		16,	17,	18,	19,		21,		23,		25,	27,	28,	29,	30,	)
		]]
		write = [32,26]
		fade = []

		arco=0

		copias=[]
		copia_fin=[]
		formula_copia=[] #No modificar
		for c in copias: #No modificar
			formula_copia.append(self.formulas[paso-1][c].copy()) #No modificar
		#Inicia escena------------------------
		self.add(self.formulas[paso-1])
		for pre_ind,post_ind in cambios:
			self.play(*[
				ReplacementTransform(
					self.formulas[paso-1][i],self.formulas[paso][j],
					path_arc=arco
					)
				for i,j in zip(pre_ind,post_ind)
				],
				*[FadeOut(self.formulas[paso-1][f])for f in fade],
				#*[Write(self.formulas[paso][w])for w in write],
				*[ReplacementTransform(formula_copia[j],self.formulas[paso][f])
				for j,f in zip(range(len(copia_fin)),copia_fin)
				],
				run_time=tiempo
			)
			self.wait(0.5)
		self.play(*[Write(self.formulas[paso][w])for w in write])
		#self.play(*[FadeOut(self.formulas[paso-1][f])for f in fade])
		self.wait(tiempo_e)

	def paso_9(self,tiempo_e):
		#Parámetros
		paso=9
		cambios = [[
			(	0,	1,	2,		4,	5,	6,	7,		9,		11,	12,		14,		16,	17,	18,	19,		21,		23,		25,	26,	27,	28,	29,	30,		32,	),
			(	0,	1,	2,		4,	5,	6,	7,		9,		11,	12,		14,		16,	21,	22,	23,		25,		17,		18,	19,	20,	21,	22,	23,		25,	)
		]]
		write = []
		fade = []

		arco=0

		copias=[]
		copia_fin=[]
		formula_copia=[] #No modificar
		for c in copias: #No modificar
			formula_copia.append(self.formulas[paso-1][c].copy()) #No modificar
		#Inicia escena------------------------
		self.add(self.formulas[paso-1])
		for pre_ind,post_ind in cambios:
			self.play(*[
				ReplacementTransform(
					self.formulas[paso-1][i],self.formulas[paso][j],
					path_arc=arco
					)
				for i,j in zip(pre_ind,post_ind)
				],
				*[FadeOut(self.formulas[paso-1][f])for f in fade],
				*[Write(self.formulas[paso][w])for w in write],
				*[ReplacementTransform(formula_copia[j],self.formulas[paso][f])
				for j,f in zip(range(len(copia_fin)),copia_fin)
				],
				run_time=tiempo
			)
			self.wait(tiempo_e)
		
	def paso_10(self,tiempo_e):
		#Parámetros
		paso=10
		cambios = [[
			(	0,	1,	2,		4,	5,	6,	7,		9,		11,	12,		14,		16,	17,	18,	19,	20,	21,	22,	23,		25,	),
			(	2,	3,	5,		6,	7,	8,	10,		12,		14,	15,		21,		22,	23,	24,	25,	26,	27,	30,	31,		32,	)
		]]
		write = [0,	1,	16,	18,	20]
		fade = []

		arco=0

		copias=[]
		copia_fin=[]
		formula_copia=[] #No modificar
		for c in copias: #No modificar
			formula_copia.append(self.formulas[paso-1][c].copy()) #No modificar

		#Parametros extra
		self.formulas[paso][30:]
		#Inicia escena------------------------
		self.add(self.formulas[paso-1])
		for pre_ind,post_ind in cambios:
			self.play(*[
				ReplacementTransform(
					self.formulas[paso-1][i],self.formulas[paso][j],
					path_arc=arco
					)
				for i,j in zip(pre_ind,post_ind)
				],
				*[FadeOut(self.formulas[paso-1][f])for f in fade],
				#*[Write(self.formulas[paso][w])for w in write],
				*[ReplacementTransform(formula_copia[j],self.formulas[paso][f])
				for j,f in zip(range(len(copia_fin)),copia_fin)
				],
				run_time=tiempo
			)
			self.wait()
		self.play(*[Write(self.formulas[paso][w])for w in write])
		#self.play(*[FadeOut(self.formulas[paso-1][f])for f in fade])
		self.wait(tiempo_e)

	def paso_11(self,tiempo_e):
		#Parámetros
		paso=11
		cambios = [[
			(				3,		5,	6,	7,	8,		10,					15,	16,		18,		20,	21,	22,	23,	24,	25,	26,	27,			30,	31,	32,	),
			(				0,		1,	3,	4,	5,		6,					8,	9,		10,		12,	14,	15,	16,	17,	18,	19,	20,			21,	24,	25,	)
		]]
		write = []
		fade = [0,	1,	2,	12,	14]

		arco=0

		copias=[]
		copia_fin=[]
		formula_copia=[] #No modificar
		for c in copias: #No modificar
			formula_copia.append(self.formulas[paso-1][c].copy()) #No modificar
		#Parametros extra
		#self.formulas[paso][].shift(RIGHT*0.2)
		#Inicia escena------------------------
		self.add(self.formulas[paso-1])
		for pre_ind,post_ind in cambios:
			self.play(*[
				ReplacementTransform(
					self.formulas[paso-1][i],self.formulas[paso][j],
					path_arc=arco
					)
				for i,j in zip(pre_ind,post_ind)
				],
				*[FadeOut(self.formulas[paso-1][f])for f in fade],
				*[Write(self.formulas[paso][w])for w in write],
				*[ReplacementTransform(formula_copia[j],self.formulas[paso][f])
				for j,f in zip(range(len(copia_fin)),copia_fin)
				],
				run_time=tiempo
			)
			self.wait(tiempo_e)
		
	def paso_12(self,tiempo_e):
		#Parámetros
		paso=12
		cambios = [[
			(	0,	1,		3,	4,	5,	6,		8,	9,	10,		12,		14,	15,	16,	17,	18,	19,	20,	21,			24,	25,	),
			(	0,	2,		4,	5,	6,	7,		1,	9,	10,		12,		14,	15,	16,	17,	18,	19,	20,	21,			24,	25,	)
		]]
		write = []
		fade = []

		arco=0

		copias=[]
		copia_fin=[]
		formula_copia=[] #No modificar
		for c in copias: #No modificar
			formula_copia.append(self.formulas[paso-1][c].copy()) #No modificar
		#Parametros extra
		#self.formulas[paso-1][21:].shift(RIGHT*0.2)
		#self.formulas[paso][21:].shift(RIGHT*0.2)
		#Inicia escena------------------------
		self.add(self.formulas[paso-1])
		for pre_ind,post_ind in cambios:
			self.play(*[
				ReplacementTransform(
					self.formulas[paso-1][i],self.formulas[paso][j],
					path_arc=arco
					)
				for i,j in zip(pre_ind,post_ind)
				],
				*[FadeOut(self.formulas[paso-1][f])for f in fade],
				*[Write(self.formulas[paso][w])for w in write],
				*[ReplacementTransform(formula_copia[j],self.formulas[paso][f])
				for j,f in zip(range(len(copia_fin)),copia_fin)
				],
				run_time=tiempo
			)
			self.wait(tiempo_e)

	def paso_13(self,tiempo_e):
		#Parámetros
		paso=13
		cambios = [[
			(	0,	1,	2,		4,	5,	6,	7,		9,	10,		12,		14,	15,	16,	17,	18,	19,	20,	21,			24,	),
			(	0,	1,	2,		4,	5,	6,	7,		9,	11,		12,		14,	15,	16,	17,	18,	20,	21,	22,			23,	)
		]]
		write = []
		fade = [25]

		arco=0

		copias=[]
		copia_fin=[]
		formula_copia=[] #No modificar
		for c in copias: #No modificar
			formula_copia.append(self.formulas[paso-1][c].copy()) #No modificar
		#Parametros extra
		#self.formulas[paso][].shift(RIGHT*0.2)
		#Inicia escena------------------------
		self.add(self.formulas[paso-1])
		for pre_ind,post_ind in cambios:
			self.play(*[
				ReplacementTransform(
					self.formulas[paso-1][i],self.formulas[paso][j],
					path_arc=arco
					)
				for i,j in zip(pre_ind,post_ind)
				],
				*[FadeOut(self.formulas[paso-1][f])for f in fade],
				*[Write(self.formulas[paso][w])for w in write],
				*[ReplacementTransform(formula_copia[j],self.formulas[paso][f])
				for j,f in zip(range(len(copia_fin)),copia_fin)
				],
				run_time=tiempo
			)
			self.wait(tiempo_e)

	def paso_14(self,tiempo_e):
		#Parámetros
		paso=14
		cambios = [[
			(	0,	1,	2,		4,	5,	6,	7,		9,		11,	12,		14,	15,	16,	17,	18,		20,	21,	22,	23,	),
			(	0,	1,	3,		4,	16,	17,	18,		5,		6,	7,		9,	10,	11,	12,	13,		15,	16,	17,	18,	)
		]]
		write = []
		fade = []

		arco=0

		copias=[]
		copia_fin=[]
		formula_copia=[] #No modificar
		for c in copias: #No modificar
			formula_copia.append(self.formulas[paso-1][c].copy()) #No modificar
		#Parametros extra
		#self.formulas[paso][].shift(RIGHT*0.2)
		#Inicia escena------------------------
		self.add(self.formulas[paso-1])
		for pre_ind,post_ind in cambios:
			self.play(*[
				ReplacementTransform(
					self.formulas[paso-1][i],self.formulas[paso][j],
					path_arc=arco
					)
				for i,j in zip(pre_ind,post_ind)
				],
				*[FadeOut(self.formulas[paso-1][f])for f in fade],
				*[Write(self.formulas[paso][w])for w in write],
				*[ReplacementTransform(formula_copia[j],self.formulas[paso][f])
				for j,f in zip(range(len(copia_fin)),copia_fin)
				],
				run_time=tiempo
			)
			self.wait(tiempo_e)
		