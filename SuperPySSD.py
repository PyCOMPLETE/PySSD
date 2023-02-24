import os
import numpy as np
from matplotlib import pyplot as plt

from cpymad.madx import Madx

import xpart as xp
import xobjects as xo
import xtrack as xt
import xfields as xf
context = xo.ContextCpu(omp_num_threads=0)

bunch_len = 0.08
energy = 7000.0
emit_norm_x = 2.5E-6
emit_norm_y = 2.5E-6
intensity = 2.2E11
oct_current = 10.0

mad = Madx()

mad.call('/afs/cern.ch/eng/lhc/optics/runIII/lhc.seq')
mad.call('/afs/cern.ch/eng/lhc/optics/HLLHCV1.5/toolkit/macro.madx')
mad.call('/afs/cern.ch/eng/lhc/optics/HLLHCV1.5/hllhc_sequence.madx')
mad.input('mylhcbeam = 1;')
mad.input(f'''
NRJ = {energy};
gamma_rel := NRJ/pmass;
emittance_norm_x = {emit_norm_x};
emittance_norm_y = {emit_norm_x};
bunch_len={bunch_len};
epsx:=emittance_norm_x /gamma_rel;
epsy:=emittance_norm_y /gamma_rel;
if (mylhcbeam>2){{ bv_aux=-1; }} else {{ bv_aux=1; }};
Beam,particle=proton,sequence=lhcb1,energy=NRJ,NPART={intensity},sige=4.5e-4*sqrt(450./NRJ),ex:=epsx,ey:=epsy,sigt:=bunch_len;
Beam,particle=proton,sequence=lhcb2,energy=NRJ,bv = -bv_aux,NPART={intensity},sige=4.5e-4*sqrt(450./NRJ),ex:=epsx,ey:=epsy,sigt:=bunch_len;
b_t_dist = 25;                  !--- bunch distance in [ns]
beam%lhcb1->sigt = bunch_len;
beam%lhcb2->sigt = bunch_len;
''')
mad.input('exec,myslice;')
mad.call('/afs/cern.ch/eng/lhc/optics/HLLHCV1.5/flatcc/opt_flathv_500_1000_thin.madx')
mad.input('''
seqedit,sequence=lhcb1;
flatten;
endedit;
seqedit,sequence=lhcb1;
cycle,start=ip3;
endedit;
seqedit,sequence=lhcb2;
flatten;
endedit;
seqedit,sequence=lhcb2;
cycle,start=ip3;
endedit;
''')
mad.input("VRF400:=16.;LAGRF400.B1=0.5;LAGRF400.B2=0.;")
mad.call("/afs/cern.ch/eng/lhc/optics/HLLHCV1.5/toolkit/rematch_chroma.madx")
mad.call("/afs/cern.ch/eng/lhc/optics/HLLHCV1.5/toolkit/rematch_tune.madx")

mad.input(f'''I_MO={oct_current};
I_MOF:=I_MO;
I_MOD:=I_MO;
brho:=NRJ*1e9/clight;
KOF.A12B1:=Kmax_MO*I_MOF/Imax_MO/brho; KOF.A23B1:=Kmax_MO*I_MOF/Imax_MO/brho;
KOF.A34B1:=Kmax_MO*I_MOF/Imax_MO/brho; KOF.A45B1:=Kmax_MO*I_MOF/Imax_MO/brho;
KOF.A56B1:=Kmax_MO*I_MOF/Imax_MO/brho; KOF.A67B1:=Kmax_MO*I_MOF/Imax_MO/brho;
KOF.A78B1:=Kmax_MO*I_MOF/Imax_MO/brho; KOF.A81B1:=Kmax_MO*I_MOF/Imax_MO/brho;
KOD.A12B1:=Kmax_MO*I_MOD/Imax_MO/brho; KOD.A23B1:=Kmax_MO*I_MOD/Imax_MO/brho;
KOD.A34B1:=Kmax_MO*I_MOD/Imax_MO/brho; KOD.A45B1:=Kmax_MO*I_MOD/Imax_MO/brho;
KOD.A56B1:=Kmax_MO*I_MOD/Imax_MO/brho; KOD.A67B1:=Kmax_MO*I_MOD/Imax_MO/brho;
KOD.A78B1:=Kmax_MO*I_MOD/Imax_MO/brho; KOD.A81B1:=Kmax_MO*I_MOD/Imax_MO/brho;
''')

mad.input(f'on_x1 = 0;')
mad.input(f'on_x5 = 0;')
mad.input(f'on_x2 = 0;')
mad.input(f'on_x8 = 0;')
mad.input(f'crab_angle = 0;')

mad.input(f'on_sep1h = 0;')
mad.input(f'on_sep1v = 0;')
mad.input(f'on_sep5h = 0;')
mad.input(f'on_sep5v = 0;')
mad.input(f'on_sep2h = 0;')
mad.input(f'on_sep2v = 0;')
mad.input(f'on_sep8h = 0;')
mad.input(f'on_sep8v = 0;')

mad.input(f'I_MO = 0;')

mad.input('''
on_alice:=1;
on_lhcb :=1;
''')

mad.input('use,sequence=lhcb1')
twiss_b1 = mad.twiss(sequence='lhcb1')
mad.input('use,sequence=lhcb2')
twiss_b2 = mad.twiss(sequence='lhcb2')

tune_x_0 = twiss_b1.summary['Q1']
tune_y_0 = twiss_b1.summary['Q2']

print(twiss_b1.name)
i_IP1 = np.where(twiss_b1.name == 'ip1:1')
i_IP2 = np.where(twiss_b1.name == 'ip2:1')
i_IP5 = np.where(twiss_b1.name == 'ip5:1')
i_IP8 = np.where(twiss_b1.name == 'ip8:1')
print('IP1',twiss_b1.betx[i_IP1],twiss_b1.bety[i_IP1],twiss_b1.x[i_IP1],twiss_b1.y[i_IP1],twiss_b1.px[i_IP1],twiss_b1.py[i_IP1])
print('IP5',twiss_b1.betx[i_IP5],twiss_b1.bety[i_IP5],twiss_b1.x[i_IP5],twiss_b1.y[i_IP5],twiss_b1.px[i_IP1],twiss_b1.py[i_IP5])
print('IP2',twiss_b1.betx[i_IP2],twiss_b1.bety[i_IP2],twiss_b1.x[i_IP2],twiss_b1.y[i_IP2],twiss_b1.px[i_IP1],twiss_b1.py[i_IP2])
print('IP8',twiss_b1.betx[i_IP8],twiss_b1.bety[i_IP8],twiss_b1.x[i_IP8],twiss_b1.y[i_IP8],twiss_b1.px[i_IP1],twiss_b1.py[i_IP8])

if False:
    plt.figure(0)
    plt.plot(twiss_b1.s, twiss_b1.betx,'-b')
    plt.plot(twiss_b1.s, twiss_b1.bety,'--b')
    plt.plot(twiss_b2.s, twiss_b2.betx,'-r')
    plt.plot(twiss_b2.s, twiss_b2.bety,'--r')
    plt.figure(1)
    plt.plot(twiss_b1.s, twiss_b1.x,'-b')
    plt.plot(twiss_b1.s, twiss_b1.y,'--b')
    plt.plot(twiss_b2.s, twiss_b2.x,'-r')
    plt.plot(twiss_b2.s, twiss_b2.y,'--r')

mad.input('''
on_ho1  := +0;
on_ho2  := +0;
on_ho5  := +1;
on_ho8  := +0;

on_lr1l := +0;
on_lr1r := +0;
on_lr2l := +0;
on_lr2r := +0;
on_lr5l := +0;
on_lr5r := +0;
on_lr8l := +0;
on_lr8r := +0;
''')

if not os.path.exists('temp'):
    os.mkdir('./temp')
if not os.path.exists('slhc'):
    os.symlink('/afs/cern.ch/eng/lhc/optics/HLLHCV1.3','slhc')

mad.call('/afs/cern.ch/eng/lhc/optics/HLLHCV1.3/beambeam/macro_bb.madx')
mad.input('''
n_insideD1 = 5;    !default value for the number of additionnal parasitic encounters inside D1

nho_IR1= 21;        ! number of slices for head-on in IR1 (between 0 and 201)
nho_IR2= 21;        ! number of slices for head-on in IR2 (between 0 and 201)
nho_IR5= 21;        ! number of slices for head-on in IR5 (between 0 and 201)
nho_IR8= 21;        ! number of slices for head-on in IR8 (between 0 and 201)

Option, echo,info,warn;
exec DEFINE_BB_PARAM;  !Define main beam-beam parameters

!Record the nominal IP position and crossing angle
if(mylhcbeam==1) {use,  sequence=lhcb1;};
if(mylhcbeam>1) {use,  sequence=lhcb2;};
twiss;

xnom1=table(twiss,IP1,x);pxnom1=table(twiss,IP1,px);ynom1=table(twiss,IP1,y);pynom1=table(twiss,IP1,py);
xnom2=table(twiss,IP2,x);pxnom2=table(twiss,IP2,px);ynom2=table(twiss,IP2,y);pynom2=table(twiss,IP2,py);
xnom5=table(twiss,IP5,x);pxnom5=table(twiss,IP5,px);ynom5=table(twiss,IP5,y);pynom5=table(twiss,IP5,py);
xnom8=table(twiss,IP8,x);pxnom8=table(twiss,IP8,px);ynom8=table(twiss,IP8,y);pynom8=table(twiss,IP8,py);
value,xnom1,xnom2,xnom5,xnom8;
value,ynom1,ynom2,ynom5,ynom8;
value,pxnom1,pxnom2,pxnom5,pxnom8;
value,pynom1,pynom2,pynom5,pynom8;

!Install b-b marker
exec INSTALL_BB_MARK(b1);exec INSTALL_BB_MARK(b2);

!Define bb lenses for both beams in all IR's and calculate # of encounters before D1
exec CALCULATE_BB_LENS;

!Install bb lenses
npara_1 = npara0_1 + n_insideD1;
npara_5 = npara0_5 + n_insideD1;
npara_2 = npara0_2 + n_insideD1;
npara_8 = npara0_8 + n_insideD1;
if(mylhcbeam==1) {exec INSTALL_BB_LENS(b1);};
if(mylhcbeam>1) {exec INSTALL_BB_LENS(b2);};

!Print the lenses in bb_lenses.dat
exec, PRINT_BB_LENSES;

ON_BB_CHARGE := 0; !Switch off the charge the bb lenses
''')
print('Installed beam-beam lenses, installing CC')
#Install Crab Cavities for the weak beam
mad.call("/afs/cern.ch/eng/lhc/optics/HLLHCV1.5/toolkit/enable_crabcavities.madx")
print('Making some twiss and removing markers')
mad.input('''
use,sequence=lhcb1;
select,flag=twiss,clear;
select,flag=twiss,class=marker,pattern=PAR.*L1,range=mbxf.4l1..4/IP1.L1,column=s,name,betx,bety,alfx,alfy,mux,muy,x,y,px,py;
select,flag=twiss,class=marker,pattern=PAR.*L5,range=mbxf.4l5..4/IP5,column=s,name,betx,bety,alfx,alfy,mux,muy,x,y,px,py;
select,flag=twiss,class=marker,pattern=PAR.*R1,range=IP1/mbxf.4r1..1,column=s,name,betx,bety,alfx,alfy,mux,muy,x,y,px,py;
select,flag=twiss,class=marker,pattern=PAR.*R5,range=IP5/mbxf.4r5..1,column=s,name,betx,bety,alfx,alfy,mux,muy,x,y,px,py;
select,flag=twiss,class=marker,pattern=IP1,column=s,name,betx,bety,alfx,alfy,mux,muy,x,y,px,py;
select,flag=twiss,class=marker,pattern=IP5,column=s,name,betx,bety,alfx,alfy,mux,muy,x,y,px,py;
twiss,file=twiss_bb.b1;system,"cat twiss_bb.b1";

use,sequence=lhcb2;
select,flag=twiss,clear;
select,flag=twiss,class=marker,pattern=PAR.*L1,range=mbxf.4l1..4/IP1.L1,column=s,name,betx,bety,alfx,alfy,mux,muy,x,y,px,py;
select,flag=twiss,class=marker,pattern=PAR.*L5,range=mbxf.4l5..4/IP5,column=s,name,betx,bety,alfx,alfy,mux,muy,x,y,px,py;
select,flag=twiss,class=marker,pattern=PAR.*R1,range=IP1/mbxf.4r1..1,column=s,name,betx,bety,alfx,alfy,mux,muy,x,y,px,py;
select,flag=twiss,class=marker,pattern=PAR.*R5,range=IP5/mbxf.4r5..1,column=s,name,betx,bety,alfx,alfy,mux,muy,x,y,px,py;
select,flag=twiss,class=marker,pattern=IP1,column=s,name,betx,bety,alfx,alfy,mux,muy,x,y,px,py;
select,flag=twiss,class=marker,pattern=IP5,column=s,name,betx,bety,alfx,alfy,mux,muy,x,y,px,py;
twiss,file=twiss_bb.b2;system,"cat twiss_bb.b2";

if(mylhcbeam==1) {use,sequence=lhcb1;};
if(mylhcbeam>1) {use,sequence=lhcb2;};

select,flag=twiss,clear;
select,flag=twiss,pattern=HO,class=beambeam,column=s,name,betx,bety,alfx,alfy,mux,muy,x,y,px,py;
twiss,file=twiss_bb;system,"cat twiss_bb";

!Remove bb markers
exec REMOVE_BB_MARKER;

ON_BB_CHARGE := 1;
''')
print('Done with BB')
mad.input('use,sequence=lhcb1')
twiss_BB_b1 = mad.twiss(sequence='lhcb1')
print(twiss_BB_b1.summary['Q1']-tune_x_0,twiss_BB_b1.summary['Q2']-tune_y_0)

line = xt.Line.from_madx_sequence(sequence=mad.sequence['lhcb1'],
           deferred_expressions=False, install_apertures=False,
           apply_madx_errors=False)
line.particle_ref = xp.Particles(p0c=energy*1E9,mass0=xp.PROTON_MASS_EV)

tracker = xt.Tracker(_context=context,line=line.copy())
twiss_table_xt = tracker.twiss()
beta_x = twiss_table_xt['betx'][0]
beta_y = twiss_table_xt['bety'][0]
alpha_x = twiss_table_xt['alfx'][0]
alpha_y = twiss_table_xt['alfy'][0]
print(beta_x,beta_y,alpha_x,alpha_y)
gammar = energy*1E9/xp.PROTON_MASS_EV
betar = np.sqrt(1-1/gammar**2)
sigma_x = np.sqrt(emit_norm_x*beta_x/gammar/betar)
sigma_px = np.sqrt(emit_norm_x/beta_x/gammar/betar)
sigma_y = np.sqrt(emit_norm_y*beta_y/gammar/betar)
sigma_py = np.sqrt(emit_norm_y/beta_y/gammar/betar)
if False:
    plt.figure(2)
    plt.plot(twiss_table_xt['s'],twiss_table_xt['betx'],'xb')
    plt.plot(twiss_table_xt['s'],twiss_table_xt['bety'],'xg')

xs = np.linspace(1E-3,6.0,100)
ys = np.linspace(1E-3,6.0,100)
X,Y = np.meshgrid(xs,ys)
flat_xs = np.reshape(X,np.shape(X)[0]*np.shape(X)[1])
flat_ys = np.reshape(Y,np.shape(Y)[0]*np.shape(Y)[1])

import time
time0 = time.time()
particles = xp.Particles(_context=context,
                         q0 = 1,p0c=energy*1E9,mass0=xp.PROTON_MASS_EV,
                         x=flat_xs*sigma_x,px=0.0,y=flat_ys*sigma_y,zeta=0,delta=0)

tracker.track(particles,num_turns=2048,turn_by_turn_monitor=True)
print(time.time()-time0)
plt.figure(10)
plt.plot(tracker.record_last_track.x,tracker.record_last_track.px,'x')
plt.figure(10)
plt.plot(tracker.record_last_track.y,tracker.record_last_track.py,'x')
plt.show()




