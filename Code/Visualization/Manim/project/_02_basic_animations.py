from manimlib import *

class NoAnimations(Scene):
    """
    In this scene there are no animations,
    neither self.wait(...) nor self.play(...)
    so it should not be rendered as video,
    but can be rendered as image using -s or -ps
    """
    def construct(self):
        text = Text("Hello world")
        self.add(text)
        # self.wait()
        # self.remove(text)

class BasicAnimations(Scene):
    def construct(self):
        text = Text("Hello word")
        self.play(
            Write(text)
            # FadeIn(text)
            # GrowFromCenter(text)
            # FadeInFromLarge(text, scale_factor=2)
        )

class BasicProgression(Scene):
    def construct(self):
        text = Text("Hello word")
        self.play(Write(text))
        self.wait() # 1 second by default
        self.play(FadeToColor(text,RED))
        self.wait()
        self.play(FadeOut(text))
        self.wait()

class ChangeDuration(Scene):
    def construct(self):
        self.play(
            Create(Circle()),
            run_time=3,
            rate_func=smooth
        )
        self.wait()

class ChangeDurationMultipleAnimations(Scene):
    def construct(self):
        self.play(
            Create(
                Circle(),
                run_time=3,
                rate_func=smooth
            ),
            FadeIn(
                Square(),
                run_time=2,
                rate_func=there_and_back
            ),
            GrowFromCenter(
                Triangle()
            )
        )
        self.wait()

class MoreAnimations(Scene):
    def construct(self):
        text = Text("Hello world")
        self.play(Write(text))
        self.wait()
        self.play(Rotate(text,PI/2))
        self.wait()
        self.play(Indicate(text))
        self.wait()
        self.play(FocusOn(text))
        self.wait()
        self.play(ShowCreationThenDestructionAround(text))
        self.wait()

# More here:
# https://elteoremadebeethoven.github.io/manim_3feb_docs.github.io/html/tree/animation.html

# The code listed on this page (which is mine)
# is from an old Manim version, but most have
# an equivalent in the ManimCE version.
# See the source code for more information.