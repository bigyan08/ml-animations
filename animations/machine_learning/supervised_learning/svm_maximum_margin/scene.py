"""An equal-scale student graph carries the complete SVM explanation."""
import numpy as np
import contourpy
import manimgx as m
from shared.scenes import NarratedScene
from model import X, Y, OUTLIER, linear_model, soft_model, nonlinear_data, nonlinear_model, candidate_clearance

m.config.pixel_width=1080
m.config.pixel_height=1920
m.config.frame_rate=30
m.config.background_color='#0B1020'
WHITE,MUTED,TEAL,ORANGE,GOLD,PINK='#F3F2ED','#AFBED3','#4DE0C1','#FFAF66','#FFD166','#FF819F'


def txt(s,y,size=28,color=WHITE,font='Inter'):
    o=m.Text(s,font=font,font_size=size,color=color,weight='BOLD' if font=='Space Grotesk' else 'MEDIUM')
    o.move_to([-.15,y,0])
    if o.width>5.85: raise ValueError(f'Reflow oversized copy: {s}')
    return o


def math(s,y=-1.7,size=38,color=WHITE):
    o=m.MathTex(s,font_size=size,color=color).move_to([-.15,y,0])
    if o.width>5.85: raise ValueError(f'Reflow oversized equation: {s}')
    return o


def pos(x,y):
    return np.array([-2.4+.75*x,-.15+.75*y,0.])


def segment(w,b,level=0):
    """Clip a level line to the fixed 0..6 square without changing plot scales."""
    a,c=np.asarray(w,dtype=float)
    vertices=[]
    for x in [0.,6.]:
        if abs(c)>1e-10:
            y=(level-b-a*x)/c
            if -1e-8<=y<=6+1e-8: vertices.append(np.array([x,np.clip(y,0,6)]))
    for y in [0.,6.]:
        if abs(a)>1e-10:
            x=(level-b-c*y)/a
            if -1e-8<=x<=6+1e-8: vertices.append(np.array([np.clip(x,0,6),y]))
    if len(vertices)<2:return None
    pairs=[(np.linalg.norm(u-v),u,v) for u in vertices for v in vertices]
    _,u,v=max(pairs,key=lambda p:p[0])
    if np.linalg.norm(u-v)<1e-8:return None
    return pos(*u),pos(*v)


def clipped_band(w,b,rho):
    points=[np.array(x,dtype=float) for x in [(0,0),(6,0),(6,6),(0,6)]]
    for sign in [1,-1]:
        result=[]
        for u,v in zip(points,points[1:]+points[:1]):
            du=sign*(np.dot(w,u)+b)-rho
            dv=sign*(np.dot(w,v)+b)-rho
            if du<=1e-8:result.append(u)
            if (du<0)!=(dv<0):result.append(u+(v-u)*du/(du-dv))
        points=result
    return m.Polygon(*[pos(*q) for q in points],fill_color=TEAL,fill_opacity=.10,stroke_width=0)


def boundary(w=(1,1),b=-5,rho=1,show_margin=True):
    g=m.VGroup()
    if show_margin:
        g.add(clipped_band(w,b,rho))
        for level in [-rho,rho]:
            ends=segment(w,b,level)
            if ends:g.add(m.DashedLine(*ends,color=MUTED,stroke_width=2,dash_length=.08))
    ends=segment(w,b)
    if ends:g.add(m.Line(*ends,color=GOLD,stroke_width=4))
    return g


class SVMMaximumMargin(NarratedScene):
    def panel(self,*objects):
        animations=[m.FadeOut(o) for o in self.current_panel]+[m.FadeIn(o) for o in objects]
        if animations:self.play(*animations,run_time=.35)
        self.current_panel=list(objects)

    def title(self,s):
        self.play(m.Transform(self.heading,txt(s,5.55,36,font='Space Grotesk')),run_time=.35)

    def change_boundary(self,w=(1,1),b=-5,rho=1,show_margin=True,duration=1.5):
        self.play(m.Transform(self.line,boundary(w,b,rho,show_margin)),run_time=duration)

    def construct(self):
        self.load_narration(__file__)
        self.current_panel=[]
        self.heading=txt('Which line is best?',5.55,40,font='Space Grotesk')
        grid=m.VGroup()
        for k in range(1,7):
            grid.add(m.Line(pos(0,k),pos(6,k),color='#27334A',stroke_width=1),
                     m.Line(pos(k,0),pos(k,6),color='#27334A',stroke_width=1))
            ylab=m.Text(str(k),font='Inter',font_size=26,color=MUTED).move_to(pos(0,k)+[-.3,0,0])
            xlab=m.Text(str(k),font='Inter',font_size=26,color=MUTED).move_to(pos(k,0)+[0,-.25,0])
            grid.add(ylab,xlab)
        axes=m.VGroup(m.Line(pos(0,0),pos(6,0),color=MUTED,stroke_width=2),m.Line(pos(0,0),pos(0,6),color=MUTED,stroke_width=2))
        labels=m.VGroup(txt('Practice sets',4.7,27,MUTED),txt('Study hours',-.7,27,MUTED))
        fail=txt('● Fail',-1.12,24,ORANGE).move_to([-1.4,-1.12,0])
        passed=txt('● Pass',-1.12,24,TEAL).move_to([1,-1.12,0])
        dots=[m.Dot(pos(*x),radius=.09,color=TEAL if y>0 else ORANGE) for x,y in zip(X,Y)]
        rings=m.VGroup(*[m.Circle(radius=.18,color=GOLD,stroke_width=3).move_to(pos(*X[i])) for i in [1,2]])
        self.line=boundary((1,1),-4.2,show_margin=False)
        self.begin_beat('hook')
        self.add(self.heading,grid,axes,labels,fail,passed,self.line)
        self.play(m.LaggedStart(*[m.FadeIn(d) for d in dots],lag_ratio=.12),run_time=1.3)
        self.cue('several different lines')
        self.change_boundary((1.5,.5),-5,show_margin=False,duration=2)
        self.change_boundary((1,1),-5,show_margin=False,duration=2)
        self.panel(txt('Several lines. Same labels.',-1.9,29,GOLD))
        self.finish()

        self.begin_beat('data');self.title('Two features, one label')
        self.play(m.FadeOut(self.line),run_time=.3)
        self.panel(math(r'\mathbf{x}=[x_1,x_2]',-1.8,42),txt('hours, practice sets',-2.65,27,MUTED))
        self.cue('Each dot');self.play(*[m.Indicate(d,color=WHITE,scale_factor=1.4) for d in dots],run_time=1)
        self.cue('Orange means');self.panel(math(r'y=-1\quad\text{Fail}',-1.8,40,ORANGE))
        self.play(m.Indicate(dots[1],color=ORANGE),run_time=.6)
        self.cue('Teal means');self.panel(math(r'y=+1\quad\text{Pass}',-1.8,40,TEAL))
        self.play(m.Indicate(dots[2],color=TEAL),run_time=.6)
        self.cue('These are labeled');self.panel(txt('Simplified training data',-1.9,28,MUTED))
        self.finish()

        self.begin_beat('decision');self.title('A score chooses a side')
        self.line=boundary(show_margin=False);self.play(m.FadeIn(self.line),run_time=.4)
        self.panel(math(r'f(\mathbf{x})=w_1x_1+w_2x_2+b',-1.85,38,GOLD))
        self.cue('Positive means');self.panel(txt('f > 0: Pass',-1.8,29,TEAL),txt('f < 0: Fail',-2.65,29,ORANGE))
        self.cue('Zero lies');self.panel(math(r'f(\mathbf{x})=0',-1.8,44,GOLD),txt('Decision boundary',-2.65,28,MUTED))
        self.cue('For our example');self.panel(math(r'f(\mathbf{x})=x_1+x_2-5',-1.9,42))
        self.cue('At two');self.panel(math(r'f(2,2)=2+2-5=-1',-1.85,35,ORANGE),txt('Predict Fail',-2.7,28,ORANGE));self.play(m.Indicate(dots[1],color=ORANGE),run_time=.6)
        self.cue('At three');self.panel(math(r'f(3,3)=3+3-5=+1',-1.85,35,TEAL),txt('Predict Pass',-2.7,28,TEAL));self.play(m.Indicate(dots[2],color=TEAL),run_time=.6)
        self.finish()

        self.begin_beat('margin');self.title('Make room on both sides')
        self.panel(txt('Same correctly classified points',-1.85,26,MUTED))
        self.cue('very close');self.change_boundary((1,1),-4.2,rho=.2,duration=2)
        self.panel(txt('Narrow side: little clearance',-1.85,27,ORANGE))
        self.cue('largest empty margin');self.change_boundary((1.5,.5),-5,rho=1,duration=2)
        self.cue('perpendicular');
        arrow=m.DoubleArrow(pos(2.4,2.8),pos(3,3),buff=.03,color=WHITE,stroke_width=3)
        self.play(m.GrowArrow(arrow),run_time=.6)
        self.cue('balanced');self.panel(txt('Balanced perpendicular margin',-1.85,27,TEAL));self.play(m.FadeOut(arrow),run_time=.3);self.change_boundary(duration=2)
        arrow=m.DoubleArrow(pos(2,2),pos(3,3),buff=.09,color=WHITE,stroke_width=3)
        self.play(m.GrowArrow(arrow),run_time=.6)
        self.cue('Feature scales');self.panel(txt('Equal axis scales',-1.75,28,MUTED),txt('Distances depend on feature scales',-2.65,24,MUTED))
        self.finish();self.remove(arrow)

        self.begin_beat('support');self.title('The points that matter')
        self.panel(txt('(2, 2): Fail',-1.75,29,ORANGE),txt('(3, 3): Pass',-2.6,29,TEAL))
        self.cue('two closest');self.play(m.Create(rings),*[d.animate.set_opacity(.18) for i,d in enumerate(dots) if i not in [1,2]],run_time=1.2)
        self.cue('support vectors');self.panel(txt('Support vectors',-1.9,32,GOLD))
        self.cue('Other points');self.play(*[d.animate.set_opacity(1) for i,d in enumerate(dots) if i not in [1,2]],run_time=.8)
        moving=dots[4];self.play(moving.animate.move_to(pos(4.8,5.4)),run_time=1.1);self.play(moving.animate.move_to(pos(*X[4])),run_time=1.1)
        self.cue('feature sums of four and six');self.panel(math(r'x_1+x_2=4\quad\text{and}\quad6',-1.85,35,MUTED))
        self.cue('halfway between');self.panel(math(r'x_1+x_2=5',-1.85,44,GOLD))
        self.finish()

        self.begin_beat('optimization');self.title('Find the widest margin')
        self.panel(math(r'f(\mathbf{x})=-1\quad\text{or}\quad+1',-1.85,36))
        self.cue('full margin width');self.panel(math(r'\text{Width}=\frac{2}{\|\mathbf{w}\|}',-1.8,46,GOLD))
        self.cue('both weights');self.panel(math(r'\frac{2}{\sqrt{1^2+1^2}}=\sqrt{2}',-1.85,43,GOLD))
        self.cue('minimizing');self.panel(math(r'\min_{\mathbf{w},b}\;\frac12\|\mathbf{w}\|^2',-1.85,44))
        self.cue('constraint');self.panel(math(r'y_i(\mathbf{w}^{T}\mathbf{x}_i+b)\geq1',-1.85,38),txt('Every point beyond its margin',-2.75,25,MUTED))
        self.cue('The solver');self.panel(txt('Candidate comparison',-1.75,28,GOLD),txt('Illustration, not a solver trace',-2.6,24,MUTED))
        self.change_boundary((1.5,.5),-5,rho=1,duration=1.1);self.change_boundary(duration=1.5)
        self.finish()

        self.begin_beat('soft');self.title('Allow a compromise')
        self.play(m.FadeOut(rings),run_time=.3)
        self.panel(txt('Real labels can overlap',-1.9,29,MUTED))
        unusual=m.Dot(pos(*OUTLIER),radius=.10,color=TEAL)
        highlight=m.Circle(radius=.2,color=PINK,stroke_width=3).move_to(pos(*OUTLIER))
        self.cue('Add a passing student');self.play(m.FadeIn(unusual),m.Create(highlight),run_time=.8)
        self.cue('No straight line');self.panel(txt('No perfect linear separator',-1.9,28,PINK))
        self.cue('A soft margin');self.panel(math(r'y_i f(\mathbf{x}_i)\geq1-\xi_i',-1.75,38),math(r'\xi_i\geq0',-2.65,36))
        self.cue('The objective');self.panel(math(r'\min\;\frac12\|\mathbf{w}\|^2+C\sum_i\xi_i',-1.85,38))
        self.cue('Larger C');self.panel(txt('C = 10: stronger penalty',-1.75,28,GOLD),txt('Actual soft-margin SVC fit',-2.6,24,MUTED))
        large=soft_model(10);self.change_boundary(large.coef_[0],large.intercept_[0],duration=1.3)
        self.cue('Smaller C');self.panel(txt('C = 0.2: wider margin',-1.75,28,TEAL),txt('Allows more margin violations',-2.6,25,MUTED))
        small=soft_model(.2);self.change_boundary(small.coef_[0],small.intercept_[0],duration=1.3)
        self.finish();self.remove(unusual,highlight)

        self.begin_beat('kernel');self.title('Beyond straight lines')
        self.play(*[m.FadeOut(d) for d in dots],m.FadeOut(self.line),run_time=.6)
        new_x,new_y=nonlinear_data()
        newdots=m.VGroup(*[m.Dot(pos(*x),radius=.075,color=TEAL if y>0 else ORANGE) for x,y in zip(new_x,new_y)])
        self.play(m.FadeIn(newdots),run_time=.6)
        self.panel(txt('New illustrative dataset',-1.8,27,MUTED))
        trial=m.Line(pos(0,3),pos(6,3),color=GOLD,stroke_width=4)
        self.cue('No straight line');self.play(m.Create(trial),run_time=.7);self.panel(txt('A line cannot isolate the center',-1.85,26,ORANGE))
        self.cue('A kernel measures');self.panel(txt('Compare similarity between points',-1.85,25))
        self.cue('An R B F kernel')
        model=nonlinear_model(); ticks=np.linspace(0,6,181)
        gx,gy=np.meshgrid(ticks,ticks)
        scores=model.decision_function(np.c_[gx.ravel(),gy.ravel()]).reshape(gx.shape)
        curves=m.VGroup()
        for vertices in contourpy.contour_generator(x=ticks,y=ticks,z=scores).lines(0):
            curve=m.VMobject(color=GOLD,stroke_width=4)
            curve.set_points_as_corners([pos(*q) for q in vertices]);curves.add(curve)
        self.play(m.FadeOut(trial),m.Create(curves),run_time=1.6)
        self.panel(txt('RBF: learned curved boundary',-1.85,27,TEAL))
        self.cue('Gamma controls');self.panel(math(r'K(\mathbf{x},\mathbf{z})=e^{-\gamma\|\mathbf{x}-\mathbf{z}\|^2}',-1.75,35),txt('γ: how fast similarity falls',-2.7,26,MUTED))
        self.finish();self.remove(newdots,curves)

        self.begin_beat('python');self.title('Train it in Python')
        self.line=boundary();self.play(*[m.FadeIn(d) for d in dots],m.FadeIn(self.line),m.FadeIn(rings),run_time=.6)
        self.panel(txt('from sklearn.svm import SVC',-1.9,22,GOLD,font='DejaVu Sans Mono'))
        self.cue('Choose a linear');self.panel(txt('model = SVC(', -1.65,24,font='DejaVu Sans Mono'),txt('kernel="linear", C=1000)',-2.55,22,GOLD,font='DejaVu Sans Mono'))
        self.cue('then fit');self.panel(txt('model.fit(X, y)',-1.9,27,GOLD,font='DejaVu Sans Mono'))
        self.cue('The model stores');self.panel(txt('model.support_vectors_',-1.9,23,font='DejaVu Sans Mono'))
        self.cue('Give it a student');self.panel(txt('model.predict(', -1.65,24,font='DejaVu Sans Mono'),txt('[[2.8, 2.6]])',-2.55,24,GOLD,font='DejaVu Sans Mono'))
        query=m.Dot(pos(2.8,2.6),radius=.1,color=WHITE)
        self.play(m.FadeIn(query),run_time=.4)
        self.cue('decision score');self.panel(math(r'f=2.8+2.6-5=0.4',-1.75,37,GOLD),txt('Positive → Pass',-2.65,29,TEAL))
        self.cue('not a probability');self.panel(txt('Score ≠ probability',-1.9,30,MUTED))
        self.cue('For a different pattern');self.panel(txt('SVC(kernel="rbf")',-1.75,24,GOLD,font='DejaVu Sans Mono'),txt('An alternative model choice',-2.65,25,MUTED))
        self.finish();self.remove(query)

        self.begin_beat('closing');self.title('The chosen boundary')
        self.panel(txt('Wide margin',-1.75,31,TEAL),txt('Control violations when needed',-2.65,26))
        self.play(m.Indicate(rings,color=GOLD,scale_factor=1.1),run_time=1)
        self.cue('tested on new examples');self.panel(txt('Test on new examples',-1.9,31,GOLD))
        self.finish()
