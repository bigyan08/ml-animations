"""One query and ten fixed students connect distance, voting and prediction maps."""
import numpy as np
import manimgx as m
from shared.scenes import NarratedScene
from model import X,Y,QUERY,neighbors,predict,region_map

m.config.pixel_width=1080
m.config.pixel_height=1920
m.config.frame_rate=30
m.config.background_color='#0B1020'
WHITE,MUTED,TEAL,ORANGE,GOLD,PINK='#F3F2ED','#AFBED3','#4DE0C1','#FFAF66','#FFD166','#FF819F'


def txt(s,y,size=28,color=WHITE,font='Inter'):
    o=m.Text(s,font=font,font_size=size,color=color,weight='BOLD' if font=='Space Grotesk' else 'MEDIUM')
    o.move_to([-.15,y,0])
    if o.width>5.85:raise ValueError(f'Reflow oversized copy: {s}')
    return o


def eq(s,y=-1.8,size=38,color=WHITE):
    o=m.MathTex(s,font_size=size,color=color).move_to([-.15,y,0])
    if o.width>5.85:raise ValueError(f'Reflow oversized equation: {s}')
    return o


def pos(x,y,lo=0,hi=6):
    return np.array([-2.4+(x-lo)*4.5/(hi-lo),-.15+(y-lo)*4.5/(hi-lo),0.])


def plot(lo=0,hi=6,minutes=False):
    g=m.VGroup()
    ticks=np.arange(lo,hi+.001,.5 if hi-lo==2 else 1)
    for value in ticks[1:]:
        g.add(m.Line(pos(lo,value,lo,hi),pos(hi,value,lo,hi),color='#27334A',stroke_width=1),
              m.Line(pos(value,lo,lo,hi),pos(value,hi,lo,hi),color='#27334A',stroke_width=1))
    # Zoom uses half-unit ticks so coordinates remain easy to read.
    labels=np.arange(lo,hi+.001,.5 if hi-lo==2 else 1)
    for value in labels:
        if value==lo and lo==0:continue
        ylab=m.Text(f'{value:g}',font='Inter',font_size=25,color=MUTED).move_to(pos(lo,value,lo,hi)+[-.31,0,0])
        xlab=m.Text(f'{value*60 if minutes else value:g}',font='Inter',font_size=25,color=MUTED).move_to(pos(value,lo,lo,hi)+[0,-.25,0])
        g.add(ylab,xlab)
    g.add(m.Line(pos(lo,lo,lo,hi),pos(hi,lo,lo,hi),color=MUTED,stroke_width=2),
          m.Line(pos(lo,lo,lo,hi),pos(lo,hi,lo,hi),color=MUTED,stroke_width=2))
    return g


def star(lo=0,hi=6):
    return m.Star(n=5,outer_radius=.15,inner_radius=.065,color=GOLD,fill_opacity=1,stroke_width=1).move_to(pos(*QUERY,lo,hi))


def region_image(k):
    labels=region_map(k)
    palette=np.array([[66,37,31,255],[18,59,53,255]],dtype=np.uint8)
    rgba=palette[labels][::-1].copy()
    image=m.ImageMobject(rgba).scale_to_fit_width(4.5).move_to([-.15,2.1,0])
    image.set_z_index(-2)
    return image


class KNNNearestNeighbors(NarratedScene):
    def panel(self,*objects):
        anim=[m.FadeOut(o) for o in self.current_panel]+[m.FadeIn(o) for o in objects]
        if anim:self.play(*anim,run_time=.35)
        self.current_panel=list(objects)

    def title(self,s):
        self.play(m.Transform(self.heading,txt(s,5.55,36,font='Space Grotesk')),run_time=.35)

    def reframe(self,lo,hi):
        self.clear_selection()
        target=plot(lo,hi)
        animations=[m.Transform(self.grid,target),m.Transform(self.query,star(lo,hi))]
        for d,x in zip(self.dots,X):
            visible=np.all(x>=lo)&np.all(x<=hi)
            # Invisible off-window records retire; original coordinates are kept in X.
            destination=pos(*x,lo,hi) if visible else d.get_center()
            animations.append(d.animate.move_to(destination).set_opacity(1 if visible else 0))
        self.play(*animations,run_time=1.5)
        self.lo,self.hi=lo,hi

    def clear_selection(self):
        if self.selection:self.play(m.FadeOut(self.selection),run_time=.25)
        self.selection=None

    def select(self,k,duration=1.,numbers=False):
        indices,distances=neighbors(k)
        group=m.VGroup()
        scale=4.5/(self.hi-self.lo)
        circle=m.Circle(radius=(distances[-1]+.03)*scale,color=GOLD,stroke_width=2,fill_color=GOLD,fill_opacity=.045).move_to(self.query.get_center())
        group.add(circle)
        offsets=[[0,-.38,0],[-.22,.30,0],[.27,-.20,0],[-.25,-.30,0],[.28,.28,0]]
        for rank,i in enumerate(indices):
            p=pos(*X[i],self.lo,self.hi)
            group.add(m.Line(self.query.get_center(),p,color=TEAL if Y[i] else ORANGE,stroke_width=2),
                      m.Circle(radius=.13,color=TEAL if Y[i] else ORANGE,stroke_width=2).move_to(p))
            if numbers:
                label=m.Text(str(rank+1),font='Inter',font_size=28,color=WHITE).move_to(p+offsets[rank])
                group.add(label)
        if self.selection:self.play(m.ReplacementTransform(self.selection,group),run_time=duration)
        else:self.play(m.FadeIn(group),run_time=duration)
        self.selection=group
        self.bring_to_front(*self.dots,self.query)

    def vote_panel(self,k):
        indices,_=neighbors(k);passed=int(sum(Y[indices]));failed=k-passed;label='PASS' if passed>failed else 'FAIL'
        self.panel(txt(f'K = {k}   •   {passed} Pass / {failed} Fail',-1.8,28),
                   txt(f'Prediction: {label}',-2.65,31,TEAL if passed>failed else ORANGE))

    def construct(self):
        self.load_narration(__file__)
        self.current_panel=[];self.selection=None;self.lo,self.hi=0,6
        self.heading=txt('Pass or fail?',5.55,42,font='Space Grotesk')
        self.grid=plot();self.query=star()
        self.xlabel=txt('Study hours',-.7,27,MUTED)
        self.ylabel=txt('Practice sets',4.7,27,MUTED)
        fail=txt('● Fail',-1.12,24,ORANGE).move_to([-1.8,-1.12,0])
        passed=txt('● Pass',-1.12,24,TEAL).move_to([-.15,-1.12,0])
        new=txt('★ New',-1.12,24,GOLD).move_to([1.55,-1.12,0])
        self.dots=[m.Dot(pos(*x),radius=.075,color=TEAL if y else ORANGE) for x,y in zip(X,Y)]
        self.begin_beat('hook')
        self.add(self.heading,self.grid,self.query,self.xlabel,self.ylabel,fail,passed,new)
        self.panel(eq(r'\mathbf{x}_{new}=(3,3)',-1.9,41,GOLD))
        self.cue('similar study habits');self.play(m.LaggedStart(*[m.FadeIn(d) for d in self.dots],lag_ratio=.08),run_time=1.5)
        self.cue('most similar students');self.select(3,2)
        self.finish()

        self.begin_beat('data');self.title('The student records');self.clear_selection()
        self.panel(txt('Ten illustrative records',-1.85,28,MUTED))
        self.cue('Hours studied');self.play(m.Indicate(self.xlabel,color=WHITE),run_time=.6)
        self.cue('Practice sets');self.play(m.Indicate(self.ylabel,color=WHITE),run_time=.6)
        self.cue('Orange dots');self.play(*[m.Indicate(d,color=ORANGE) for d,y in zip(self.dots,Y) if not y],run_time=.7)
        self.cue('yellow star');self.panel(eq(r'\mathbf{x}_{new}=(3,3)',-1.85,42,GOLD));self.play(m.Indicate(self.query,color=GOLD),run_time=.7)
        self.finish()

        self.begin_beat('distance');self.title('Measure similarity')
        self.cue('zoom');self.panel(txt('Zoom: axes 2–4',-1.9,27,MUTED));self.reframe(2,4)
        self.cue('Subtract');self.panel(eq(r'd=\sqrt{(x_1-z_1)^2+(x_2-z_2)^2}',-1.9,35))
        self.cue('Euclidean distance');self.panel(txt('Euclidean distance',-1.85,31,GOLD))
        self.cue('zero point two');self.select(1,.65)
        self.panel(eq(r'd=\sqrt{0.2^2+0^2}=0.20',-1.85,40,ORANGE))
        self.cue('zero point five');self.clear_selection()
        comparison=m.VGroup(m.DashedLine(self.query.get_center(),pos(*X[0],2,4),color=ORANGE),m.DashedLine(self.query.get_center(),pos(*X[1],2,4),color=TEAL))
        self.play(m.Create(comparison),run_time=.5)
        self.panel(eq(r'd=\sqrt{0.4^2+0.3^2}=0.50',-1.85,39,TEAL))
        self.cue('sort the examples');self.play(m.FadeOut(comparison),run_time=.25);self.select(3,.8,numbers=True)
        self.panel(txt('1: 0.20 Fail   •   2: 0.50 Pass',-1.75,25),txt('3: 0.63 Pass',-2.65,28,TEAL))
        self.finish()

        self.begin_beat('vote');self.title('Let three neighbors vote')
        self.panel(txt('K = number of neighbors',-1.85,29,GOLD))
        self.cue('K equals three');self.select(3,.8,numbers=True)
        self.cue('nearest student failed');self.panel(txt('Neighbor 1: Fail',-1.85,30,ORANGE))
        self.cue('next two passed');self.panel(txt('Neighbors 2 and 3: Pass',-1.85,28,TEAL))
        self.cue('two votes');self.vote_panel(3)
        self.cue('zero means fail');self.panel(txt('0 = Fail   •   1 = Pass',-1.85,29))
        self.cue('most common label');self.panel(eq(r'\hat y=\operatorname{mode}(0,1,1)=1',-1.85,39,TEAL),txt('Prediction: PASS',-2.7,30,TEAL))
        self.finish()

        self.begin_beat('change');self.title('Change K')
        self.cue('With K equal to one');self.vote_panel(1);self.select(1,1)
        self.cue('With three neighbors');self.vote_panel(3);self.select(3,1,numbers=True)
        self.cue('With five neighbors');self.vote_panel(5);self.select(5,1)
        self.cue('Nothing');self.panel(txt('Same student. Same records.',-1.75,28,MUTED),txt('Only K changes',-2.65,30,GOLD))
        self.finish()

        self.begin_beat('regions');self.title('A map of predictions')
        self.panel(txt('Full graph: axes 0–6',-1.85,28,MUTED));self.reframe(0,6)
        self.cue('every possible location')
        current=region_image(1);self.add(current);self.bring_to_back(current);self.play(m.FadeIn(current),run_time=1)
        self.panel(txt('K = 1: highly local decisions',-1.85,27,GOLD))
        self.cue('With three and five')
        for k in [3,5]:
            target=region_image(k);self.add(target);self.bring_to_back(target)
            self.play(m.FadeOut(current),m.FadeIn(target),run_time=1.2);self.remove(current);current=target
            self.panel(txt(f'K = {k}: more neighboring votes',-1.85,26,GOLD))
            self.wait(.7)
        self.cue('Small K');self.panel(txt('Small K: may overfit',-1.75,28,ORANGE),txt('Very large K: may underfit',-2.65,27,MUTED))
        self.cue('Choose K');self.panel(txt('Choose K with validation',-1.75,28,TEAL),txt('Evaluate on separate test data',-2.65,25,MUTED))
        self.finish();self.remove(current)

        self.begin_beat('scaling');self.title('Units change distance')
        self.panel(txt('Same records. Different units.',-1.85,27,MUTED))
        self.cue('minutes');self.play(m.Transform(self.grid,plot(minutes=True)),m.Transform(self.xlabel,txt('Study minutes',-.7,27,MUTED)),run_time=1)
        self.cue('same difference');self.panel(eq(r'0.4\text{ h}\longrightarrow24\text{ min}',-1.85,40,GOLD))
        self.cue('dominate');self.panel(eq(r'd^2=0.16+0.09\quad\text{(hours)}',-1.65,33),eq(r'd^2=576+0.09\quad\text{(minutes)}',-2.65,33,PINK))
        self.cue('standardization');self.panel(eq(r'z=\frac{x-\mu}{\sigma}',-1.85,51,GOLD),txt('One mean and scale per feature',-2.8,25,MUTED))
        self.cue('Fit that transformation');self.panel(txt('Fit scaler on training data only',-1.75,26,TEAL),txt('Reuse it for validation and queries',-2.65,24,MUTED))
        self.finish()

        self.begin_beat('lazy');self.title('Store now, search later')
        self.play(m.Transform(self.grid,plot()),m.Transform(self.xlabel,txt('Study hours',-.7,27,MUTED)),run_time=.7)
        self.panel(txt('Original hours restored',-1.85,26,MUTED))
        self.cue('stores the labeled');self.panel(txt('Fit: store labeled examples',-1.75,28,GOLD),txt('Search indexes are optional',-2.65,25,MUTED))
        storage=m.RoundedRectangle(width=1.25,height=.65,corner_radius=.1,color=MUTED,fill_color='#14243A',fill_opacity=1).move_to([1.5,3.8,0])
        storage_label=m.Text('Records',font='Inter',font_size=20,color=WHITE).move_to(storage.get_center())
        self.play(m.FadeIn(storage),m.FadeIn(storage_label),run_time=.5)
        self.cue('When a new student');self.select(3,1)
        self.panel(txt('Predict: search + vote',-1.85,30,TEAL))
        self.cue('lazy learning');self.panel(txt('Lazy learning',-1.85,32,GOLD))
        self.cue('more stored examples');self.panel(txt('More memory and search work',-1.85,27,MUTED))
        self.finish();self.remove(storage,storage_label)

        self.begin_beat('python');self.title('Make the prediction')
        self.panel(txt('Original unscaled data',-1.85,28,MUTED))
        self.cue('Create a K Nearest');self.panel(txt('KNeighborsClassifier(', -1.65,24,font='DejaVu Sans Mono'),txt('n_neighbors=3)',-2.55,25,GOLD,font='DejaVu Sans Mono'))
        self.cue('Fit the ten');self.panel(txt('model.fit(X, y)',-1.9,27,GOLD,font='DejaVu Sans Mono'))
        self.cue('Now predict');self.panel(txt('model.predict([[3, 3]])',-1.9,23,GOLD,font='DejaVu Sans Mono'));self.select(3,1)
        self.cue('labels zero');self.panel(eq(r'[0,1,1]\longrightarrow1',-1.8,44,TEAL),txt('Prediction: PASS',-2.7,30,TEAL))
        self.cue('practical workflow');self.panel(txt('Pipeline: scaling → KNN',-1.75,28,TEAL),txt('Choose K with validation',-2.65,27,MUTED))
        self.finish()

        self.begin_beat('closing');self.title('Nearby examples decide')
        self.panel(txt('Same student • zoom 2–4',-1.85,27,MUTED));self.reframe(2,4);self.select(3,.8)
        self.panel(txt('Distance → neighbors → predict',-1.85,27,GOLD))
        self.cue('Classification');self.panel(txt('Classification: vote',-1.75,29,TEAL))
        self.cue('Regression');self.panel(txt('Classification: vote',-1.75,29,TEAL),txt('Regression: average',-2.65,29,GOLD))
        self.cue('checked on new data');self.panel(txt('Test on new data',-1.85,31,GOLD))
        self.finish()
