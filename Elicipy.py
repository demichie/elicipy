import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("TkAgg")

import openpyxl
from pathlib import Path

import sys
import os
import re
import pkg_resources
import numpy as np
from scipy.stats import chi2
from global_weights import global_weights
from item_weights import item_weights
from script_fromR import createDATA1
from matplotlib.ticker import PercentFormatter
from scipy import stats
import difflib
from pptx import Presentation
from pptx.enum.text import MSO_AUTO_SIZE
from pptx.enum.shapes import MSO_SHAPE
from pptx.dml.color import RGBColor
from pptx.util import Inches, Pt
from pptx.enum.dml import MSO_THEME_COLOR
from difflib import SequenceMatcher
from pptx.parts.embeddedpackage import EmbeddedXlsxPart
from pptx.parts.chart import ChartPart
import copy

import datetime

analysis = True
target = True
log = 0
n_sample = 50000
n_bins = 20


path = './OUTPUT'

# Check whether the specified path exists or not
isExist = os.path.exists(path)

if not isExist:
  
  # Create a new directory because it does not exist 
  os.makedirs(path)
  print("The new directory OUTPUT is created!")

if os.path.isfile('scale_sd.txt') == True and os.path.isfile('scale_tg.txt') == True:
    sc_sda = np.loadtxt('scale_sd.txt', dtype='str')
    sc_sd = sc_sda.tolist()
    sc_tga = np.loadtxt('scale_tg.txt', dtype='str', delimiter=',')
    sc_tg = sc_tga.tolist()
    sc_all = sc_sd + sc_tg

def iter_cells(table):
    for row in table.rows:
        for cell in row.cells:
            yield cell

def copy_slide_from_external_prs(src, idx, newPrs):

    # specify the slide you want to copy the contents from
    src_slide = src.slides[idx]

    # Define the layout you want to use from your generated pptx
    SLD_LAYOUT = 6
    slide_layout = prs.slide_layouts[SLD_LAYOUT]

    # create now slide, to copy contents to
    curr_slide = newPrs.slides.add_slide(slide_layout)

    # create images dict
    imgDict = {}

    # now copy contents from external slide, but do not copy slide properties
    # e.g. slide layouts, etc., because these would produce errors, as diplicate
    # entries might be generated
    for shp in src_slide.shapes:
        if 'Picture' in shp.name:
            # save image
            with open(shp.name+'.jpg', 'wb') as f:
                f.write(shp.image.blob)

            # add image to dict
            imgDict[shp.name+'.jpg'] = [shp.left, shp.top, shp.width, shp.height]
        else:
            # create copy of elem
            el = shp.element
            newel = copy.deepcopy(el)

            # add elem to shape tree
            curr_slide.shapes._spTree.insert_element_before(newel, 'p:extLst')

    # add pictures
    for k, v in imgDict.items():
        curr_slide.shapes.add_picture(k, v[0], v[1], v[2], v[3])
        os.remove(k)


prs=Presentation()
prs.slide_width = Inches(16)
prs.slide_height = Inches(9)

lyt=prs.slide_layouts[0] # choosing a slide layout
slide=prs.slides.add_slide(lyt) # adding a slide
title=slide.shapes.title # assigning a title
subtitle=slide.placeholders[1] # placeholder for subtitle
title.text="Expert elicitation" # title
Current_Date_Formatted = datetime.datetime.today().strftime ('%d-%b-%Y')

subtitle.text = Current_Date_Formatted # subtitle

required = {'easygui','tkinter'}
installed = {pkg.key for pkg in pkg_resources.working_set}

print("Select your *.csv file for seed")

if ( 'easygui' in installed ):      
    import easygui
    filename = easygui.fileopenbox(msg='select your *.csv file', filetypes = ['*.csv'])

elif ( 'tkinter' in installed ):
    from  tkinter import *
    root = Tk()
    root.filename =  filedialog.askopenfilename(title = 'select your *.csv file',filetypes=[("csv files", "*.csv")])
    filename = root.filename
    root.destroy()

df_seed = pd.read_csv(filename)

firstname = df_seed[df_seed.columns[1]].astype(str).tolist()
surname = df_seed[df_seed.columns[2]].astype(str).tolist()

NS_seed = []

for name, surname in zip(firstname, surname):

    NS_seed.append(name+surname)


cols_as_np = df_seed[df_seed.columns[3:]].to_numpy()


n_experts = cols_as_np.shape[0]
n_pctl = 3
n_seed = int(cols_as_np.shape[1]/n_pctl)

Cal_var = np.reshape(cols_as_np,(n_experts,n_seed,n_pctl))

Cal_var = np.swapaxes(Cal_var,1,2)
Cal_var = np.sort(Cal_var, axis=1)   

seed_question = []
su_unitok=[]

for i in range(3,3+n_seed*n_pctl,3):

    string1 = df_seed.columns[i]
    match1 = string1[string1.index("[")+1:string1.index("]")]
    su_unitok.append(match1)
    string2 = df_seed.columns[i+1]
    string3 = df_seed.columns[i+2]
    match12 = SequenceMatcher(None, string1, string2).find_longest_match(0, len(string1), 0, len(string2))
    string12 = string1[match12.a: match12.a + match12.size]
    match = SequenceMatcher(None, string12, string3).find_longest_match(0, len(string12), 0, len(string3))
    
    
    seed_question.append(string12[match.a: match.a + match.size])
    print('match',string12[match.a: match.a + match.size])
    

print("Seed_units = ",su_unitok)


if os.path.isfile('scale_sd.txt') == False:
    sc_sd = []

    idx = 1
    for x in su_unitok:
        if x == "%":
           sc_sd.append("uni%")
        elif x != "%":
           if log == 0:
               sc_sd.append("uni")
           elif log == 1:
               inp = input("Enter scale (uni/log) for Seed Question %s (%s): " % (str(idx), x))
               sc_sd.append(inp)
    
        idx += 1
    np.savetxt('scale_sd.txt', sc_sd, delimiter=',', fmt='%s')

    print(sc_sd)


"""
for i in np.arange(n_experts):

    print('expert ',i)
    print(Cal_var[i,:,:])
"""

for i in np.arange(n_seed):

    for k in np.arange(n_experts):
         if Cal_var[k,0,i] == Cal_var[k,1,i]:
                Cal_var[k,0,i] = Cal_var[k,1,i]*0.99
         if Cal_var[k,2,i] == Cal_var[k,1,i]:
                Cal_var[k,2,i] = Cal_var[k,1,i]*1.01

    print('Seed question ',i)
    print(Cal_var[:,:,i])


if target:

       
    print("")
    print("Select your *.csv file for target")

    if ( 'easygui' in installed ):      
        import easygui
        filename = easygui.fileopenbox(msg='select your *.csv file', filetypes = ['*.csv'])

    elif ( 'tkinter' in installed ):
        from  tkinter import *
        root = Tk()
        root.filename =  filedialog.askopenfilename(title = 'select your *.csv file',filetypes=[("csv files", "*.csv")])
        filename = root.filename
        root.destroy()

    df_TQ = pd.read_csv(filename)

    firstname = df_TQ[df_TQ.columns[1]].astype(str).tolist()
    surname = df_TQ[df_TQ.columns[2]].astype(str).tolist()

    TQ_seed = []

    for name, surname in zip(firstname, surname):

        TQ_seed.append(name+surname)


    sorted_idx = []
    for TQ_name in TQ_seed:

        index = NS_seed.index(difflib.get_close_matches(TQ_name, NS_seed)[0])
        sorted_idx.append(index)
    
    print('Sorted list of experts to match the order of seeds:',sorted_idx)    


    cols_as_np = df_TQ[df_TQ.columns[3:]].to_numpy()

    cols_as_np = cols_as_np[sorted_idx,:]

    n_experts_TQ = cols_as_np.shape[0]

    if ( n_experts_TQ != n_experts ):

        print('Error: number of experts in seeds and targets different')
        sys.exit()

    n_pctl = 3
    n_TQ = int(cols_as_np.shape[1]/n_pctl)

    TQs = np.reshape(cols_as_np,(n_experts,n_TQ,n_pctl))

    TQs = np.swapaxes(TQs,1,2)
    TQs = np.sort(TQs, axis=1)   

    TQ_question = []
    tg_unitok=[]

    for i in range(3,3+n_TQ*n_pctl,3):

        string1 = df_TQ.columns[i][0:20]
        string2 = df_TQ.columns[i]
        match1 = string2[string2.index("[")+1:string2.index("]")]
        tg_unitok.append(match1)
        string2 = df_TQ.columns[i+1][0:20]
        string3 = df_TQ.columns[i+2][0:20]
        match12 = SequenceMatcher(None, string1, string2).find_longest_match(0, len(string1), 0, len(string2))
        string12 = string1[match12.a: match12.a + match12.size]
        match = SequenceMatcher(None, string12, string3).find_longest_match(0, len(string12), 0, len(string3))
    
        TQ_question.append(string12[match.a: match.a + match.size])
        print('match',string12[match.a: match.a + match.size])

    print("Target_units = ",tg_unitok)

    if os.path.isfile('scale_tg.txt') == False:

        sc_tg = []

        idx = 1
        for x in tg_unitok:
            if x == "%":
               sc_tg.append("uni%")
            else:
               if log == 0:
                   sc_tg.append("uni")
               elif log == 1:
                   inp = input("Enter scale (uni/log) for Target Question %s (%s): " % (str(idx), x))
                   sc_tg.append(inp)
            idx += 1

        np.savetxt('scale_tg.txt', sc_tg, delimiter = ',', fmt = '%s')

        sc_all = sc_sd + sc_tg

        print(sc_tg)

    """
    for i in np.arange(n_experts):

        print('expert ',i)
        print(TQs[i,:,:])
    """

    for i in np.arange(n_TQ):

        for k in np.arange(n_experts):
             if TQs[k,0,i] == TQs[k,1,i]:
                  TQs[k,0,i] = TQs[k,1,i]*0.99
             if TQs[k,2,i] == TQs[k,1,i]:
                  TQs[k,2,i] = TQs[k,1,i]*1.01

        print('Target question ',i)
        print(TQs[:,:,i])

else:

    n_TQ = 0
    n_pctl = 3
    
    TQs = np.zeros((n_experts,n_pctl,n_TQ))
   

print("")
print("Select your *.xlsx file for realization")

if ( 'easygui' in installed ):      
    import easygui
    filename = easygui.fileopenbox(msg='select your *.xlsx file', filetypes = ['*.xlsx'])

elif ( 'tkinter' in installed ):
    from  tkinter import *
    root = Tk()
    root.filename =  filedialog.askopenfilename(title = 'select your *.xlsx file',filetypes=[("xlsx files", "*.xlsx")])
    filename = root.filename
    root.destroy()


wb_obj = openpyxl.load_workbook(filename) 

# Read the active sheet:
sheet = wb_obj.active

i=0
a=[]
for row in sheet.iter_rows(max_row=2):
    for cell in row:
        if i==1:
            a.append(cell.value)
        
    i=i+1

if target:

    nTot = TQs.shape[2]+Cal_var.shape[2]

else:

    nTot = Cal_var.shape[2]

realization = np.zeros(TQs.shape[2]+Cal_var.shape[2])   
realization[0:Cal_var.shape[2]] = a[0:Cal_var.shape[2]]     

print("")
print('Realization',realization)

back_measure = []

for i in np.arange(TQs.shape[2]+Cal_var.shape[2]):

    back_measure.append('uni')
    

# parameters for DM
alpha = 0.05 # significance level (this value cannot be higher than the
# highest calibration score of the pool of experts)
k = 0.1      # overshoot for intrinsic range

# global cal_power
cal_power = 1 # this value should be between [0.1, 1]. The default is 1.

optimization = 'no' # choose from 'yes' or 'no'

weight_type = 'global' # choose from 'equal', 'item', 'global', 'user'

N_max_it = 5 # maximum number of seed items to be removed at a time when

if analysis:

    if optimization=='no':

        if weight_type == 'global':

            W = global_weights(Cal_var, TQs, realization, alpha, back_measure, k,cal_power)
        
        elif weight_type == 'item':
    
            [unorm_w, W_itm, W_itm_tq] = item_weights(Cal_var, TQs, realization, alpha, back_measure, k, cal_power)

    Weq=np.ones(n_experts)
    Weqok = [x/n_experts for x in Weq]

    W_gt0_01 = []
    expin = []

    for x in W[:,4]:
        if x > 0:
           W_gt0_01.append(x)

    k = 1
    for i in W[:,4]:
        if i > 0:
           expin.append(k)
        k += 1

    W_gt0 = [round((x*100), 1) for x in W_gt0_01]

    slide = prs.slides.add_slide(prs.slide_layouts[5])
    title_shape = slide.shapes.title
    title_shape.text = "Experts' weights"
    title_para = slide.shapes.title.text_frame.paragraphs[0]
    title_para.font.name = "Helvetica"
    # ---add table weights to slide---
    x, y, cx, cy = Inches(2), Inches(2), Inches(8), Inches(4)
    #x, y, cx, cy = Inches(2), Inches(2), MSO_AUTO_SIZE.SHAPE_TO_FIT_TEXT, Inches(4)
    shape = slide.shapes.add_table(2, len(W_gt0)+1, x, y, cx, MSO_AUTO_SIZE.SHAPE_TO_FIT_TEXT)
    #shape = slide.shapes.add_table(2, len(W_gt0)+1, x, y, cx, cy)
    table = shape.table

    cell = table.cell(0, 0)
    cell.text = 'Expert ID'

    cell = table.cell(1, 0)
    cell.text = 'Expert weight'

    for j in np.arange(len(W_gt0)):
        cell = table.cell(0, j+1)
        cell.text = 'Exp'+str(expin[j])

        cell = table.cell(1, j+1)
        cell.text = '%6.2f' % W_gt0[j]

    for cell in iter_cells(table):
        for paragraph in cell.text_frame.paragraphs:
            for run in paragraph.runs:
                run.font.size = Pt(12)

    print("")
    print('W')
    print(W[:,-1])
    print("")
    print('Weq')
    print(Weqok)

DAT = np.zeros((n_experts*(n_seed+n_TQ),n_pctl+2))


DAT[:,0] = np.repeat(np.arange(1,n_experts+1),n_seed+n_TQ)
DAT[:,1] = np.tile(np.arange(1,n_seed+n_TQ+1),n_experts)

DAT[:,2:] = np.append(Cal_var,TQs,axis=2).transpose(0,2,1).reshape(-1,3)


q05 = []
q50 = []
q95 = []

q05_EW = []
q50_EW = []
q95_EW = []

figs_h={}
axs_h={}
axs_h2={}

plt.rcParams.update({'font.size': 8})


print("")
if analysis:
    print("j,quan05,quan50,qmean,quan95")

for j in np.arange(n_seed+n_TQ):

    if analysis:

        if sc_all[j] == "uni%":

            quan05,quan50,qmean,quan95,C = createDATA1(DAT,j,W[:,4].flatten(),n_sample,'red',10,60,False,'',0,0,[0,100],1)
            quan05_EW,quan50_EW,qmean_EW,quan95_EW,C_EW = createDATA1(DAT,j,Weqok,n_sample,'green',10,60,False,'',0,0,[0,100],1)

        elif sc_all[j] == "uni":

            quan05,quan50,qmean,quan95,C = createDATA1(DAT,j,W[:,4].flatten(),n_sample,'red',10,60,False,'',0,0,[0,np.inf],1)
            quan05_EW,quan50_EW,qmean_EW,quan95_EW,C_EW = createDATA1(DAT,j,Weqok,n_sample,'green',10,60,False,'',0,0,[0,np.inf],1)

        else:
     
            quan05,quan50,qmean,quan95,C = createDATA1(DAT,j,W[:,4].flatten(),n_sample,'red',10,60,True,'',0,0,[-np.inf,np.inf],1)
            quan05_EW,quan50_EW,qmean_EW,quan95_EW,C_EW = createDATA1(DAT,j,Weqok,n_sample,'green',10,60,True,'',0,0,[-np.inf,np.inf],1)
    
        print(j,quan05,quan50,qmean,quan95)
        print(j,quan05_EW,quan50_EW,qmean_EW,quan95_EW)
        
        q05.append(quan05)
        q50.append(quan50)
        q95.append(quan95)
    
        q05_EW.append(quan05_EW)
        q50_EW.append(quan50_EW)
        q95_EW.append(quan95_EW)

    
        if ( j>=n_seed):

            ntarget = str(j-n_seed+1)
        
            figs_h[j] = plt.figure()
            axs_h[j] = figs_h[j].add_subplot(111)
            C_stack = np.stack((C,C_EW), axis=0)
            wg = np.ones_like(C_stack.T) / n_sample
        
            axs_h[j].hist(C_stack.T,bins=n_bins,weights=wg,rwidth=0.5, color = ['orange','springgreen'])
                
            axs_h[j].set_xlabel(tg_unitok[j-n_seed])
            plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
        
        
            xt = plt.xticks()[0]
            
            axs_h2[j] = axs_h[j].twinx()
                    
            gkde = stats.gaussian_kde(C)
            gkde_EW = stats.gaussian_kde(C_EW)
           
            if sc_all[j] == "uni%":
            
                xmin = 0.0
                xmax = 100.0
                              
            elif sc_all[j] == "uni":
            
                xmin = 0.0
                xmax = np.amax(C_stack)
                              
            else: 

                xmin = np.amin(C_stack)
                xmax = np.amax(C_stack)
                            
            gkde_norm = gkde.integrate_box_1d(xmin,xmax)
            gkde_EW_norm = gkde_EW.integrate_box_1d(xmin,xmax)
            
            lnspc = np.linspace(xmin, xmax,1000)
            kdepdf = gkde.evaluate(lnspc) / gkde_norm
            kdepdf_EW = gkde_EW.evaluate(lnspc) / gkde_EW_norm
            axs_h2[j].plot(lnspc, kdepdf, 'r--')
            axs_h2[j].plot(lnspc, kdepdf_EW, 'g--')
            
            axs_h[j].set_xlim(xmin,xmax)
            axs_h2[j].set_xlim(xmin,xmax)    
                        
            axs_h2[j].set_ylabel('PDF', color='b')

            axs_h2[j].set_ylim(bottom=0)
            plt.legend(['CM', 'EW'])
            plt.title('Target Question '+str(j-n_seed+1))
        
            figname = path+'/hist_'+str(j-n_seed+1).zfill(2)+'.pdf'
            figs_h[j].savefig(figname)
        
            figname = path+'/hist_'+str(j-n_seed+1).zfill(2)+'.png'
            figs_h[j].savefig(figname, dpi=200)
        
            blank_slide_layout = prs.slide_layouts[6]
            title_slide_layout = prs.slide_layouts[5]
            slide = prs.slides.add_slide(title_slide_layout)
            left=Inches(2)
            top=Inches(1.5)
        
            title_shape = slide.shapes.title
            title_shape.text = TQ_question[j-n_seed]
            title_para = slide.shapes.title.text_frame.paragraphs[0]
            title_para.font.name = "Helvetica"
    
    
            img=slide.shapes.add_picture('./'+figname,left,top,width=Inches(10))
            shape = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, Inches(0.2),Inches(16),Inches(0.3))
            shape.shadow.inherit = False
            fill=shape.fill
            fill.solid()
            fill.fore_color.rgb=RGBColor(0,0,255)
            shape.text= "Expert elicitation " + datetime.datetime.today().strftime ('%d-%b-%Y')
            shape_para = shape.text_frame.paragraphs[0]
            shape_para.font.name = "Helvetica"
            plt.close()


if analysis:
    slide = prs.slides.add_slide(prs.slide_layouts[5])
    title_shape = slide.shapes.title
    title_shape.text = "Percentiles of target questions"
    title_para = slide.shapes.title.text_frame.paragraphs[0]
    title_para.font.name = "Helvetica"
    # ---add table to slide---
    x, y, cx, cy = Inches(2), Inches(2), Inches(12), Inches(4)
    shape = slide.shapes.add_table(n_TQ+1, 4, x, y, cx, MSO_AUTO_SIZE.SHAPE_TO_FIT_TEXT)
    table = shape.table

    cell = table.cell(0, 1)
    cell.text = 'Q05'

    cell = table.cell(0, 2)
    cell.text = 'Q50'

    cell = table.cell(0, 3)
    cell.text = 'Q95'

    for j in np.arange(n_TQ):

        cell = table.cell(j+1, 0)
        cell.text = 'Target Question '+str(j+1)
    
        cell = table.cell(j+1, 1)
        cell.text = '%6.2f' % q05[n_seed+j]

        cell = table.cell(j+1, 2)
        cell.text = '%6.2f' % q50[n_seed+j]

        cell = table.cell(j+1, 3)
        cell.text = '%6.2f' % q95[n_seed+j]

    for cell in iter_cells(table):
        for paragraph in cell.text_frame.paragraphs:
            for run in paragraph.runs:
                run.font.size = Pt(14)

h = 0
figs0={}
axs0={}
  
    
for j in np.arange(n_seed):

    nseed = str(j+1)

    x = Cal_var[:,1,j]
    y = np.arange(n_experts)+1
  
    # creating error
    x_errormax = Cal_var[:,2,j]-Cal_var[:,1,j]
    x_errormin = Cal_var[:,1,j]-Cal_var[:,0,j]
  
    x_error = [x_errormin, x_errormax]
  
    figs0[j] = plt.figure()
    axs0[j] = figs0[j].add_subplot(111)
    axs0[j].errorbar(x, y,xerr=x_error,fmt='bo') 
    axs0[j].plot(x-x_errormin,y,'bx')
    axs0[j].plot(x+x_errormax,y,'bx')

    if analysis:
    
        axs0[j].errorbar([q50[h]], n_experts+1,xerr=[[q50[h]-q05[h]],[q95[h]-q50[h]]],fmt='ro') 
        axs0[j].plot(q05[h],n_experts+1,'rx')
        axs0[j].plot(q95[h],n_experts+1,'rx')

        axs0[j].errorbar([q50_EW[h]], n_experts+2,xerr=[[q50_EW[h]-q05_EW[h]],[q95_EW[h]-q50_EW[h]]],fmt='go') 
        axs0[j].plot(q05_EW[h],n_experts+2,'gx')
        axs0[j].plot(q95_EW[h],n_experts+2,'gx')
    
        axs0[j].plot(realization[j],n_experts+3,'kx')

    else:
    
        axs0[j].plot(realization[j],n_experts+1,'kx')

    xt = plt.xticks()[0]  
    xmin, xmax = min(xt), max(xt)

    if ( realization[j] > 999 ):
        txt = '%5.2e' % realization[j]
    else: 
        txt = '%6.2f' % realization[j]
            
    if analysis: 

        b = np.amin([np.amin(Cal_var[:,0,j]),q05[h],realization[j]])
        c = np.amin([np.amax(Cal_var[:,0,j]),q95[h],realization[j]])  
        axs0[j].annotate(txt, (realization[j], n_experts+3+0.15))
        
    else:
    
        b = np.amin([np.amin(Cal_var[:,0,j]),realization[j]])
        c = np.amin([np.amax(Cal_var[:,0,j]),realization[j]])  
        axs0[j].annotate(txt, (realization[j], n_experts+1+0.15))
        
    
    ytick = []             
    for i in y:
        ytick.append('Exp.'+str(int(i)))

    if analysis:
    
        ytick.append('DM-Cooke')
    
        ytick.append('DM-Equal')
    
    ytick.append('Realization')
                 
    ytick_tuple = tuple(i for i in ytick)

    if analysis:
    
        y = np.arange(n_experts+3)+1

    else:

        y = np.arange(n_experts+1)+1

    axs0[j].set_yticks(y)
    axs0[j].set_yticklabels(ytick_tuple)
    axs0[j].set_xlabel(su_unitok[j])
    plt.title('Seed Question '+str(j+1))
    
    if (np.abs(c-b) >= 9.99e2):
    
        axs0[j].set_xscale('log')
    
    if analysis:    
        axs0[j].set_ylim(0.5,n_experts+4.0)
    else:
        axs0[j].set_ylim(0.5,n_experts+2.0)


    axs0[j].grid()

    figname = path+'/seed_'+str(j+1).zfill(2)+'.pdf'
    figs0[j].savefig(figname)

    figname = path+'/seed_'+str(j+1).zfill(2)+'.png'
    figs0[j].savefig(figname,dpi=200)

    title_slide_layout = prs.slide_layouts[5]
    left=Inches(2)
    top=Inches(1.5)
        
    slide = prs.slides.add_slide(title_slide_layout)
    
    title_shape = slide.shapes.title
    title_shape.text = seed_question[j]
    title_para = slide.shapes.title.text_frame.paragraphs[0]

    title_para.font.name = "Helvetica"
    
    img=slide.shapes.add_picture('./'+figname,left,top,width=Inches(10))
    
    shape = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, Inches(0.2),Inches(16),Inches(0.3))
    shape.shadow.inherit = False
    fill=shape.fill
    fill.solid()
    fill.fore_color.rgb=RGBColor(0,0,255)
    shape.text= "Expert elicitation " + datetime.datetime.today().strftime ('%d-%b-%Y')
    shape_para = shape.text_frame.paragraphs[0]
    shape_para.font.name = "Helvetica"
    plt.close()
    
    h = h+1


figs={}
axs={}
    
for j in np.arange(n_TQ):

    x = TQs[:,1,j]
    y = np.arange(n_experts)+1
  
    # creating error
    x_errormax = TQs[:,2,j]-TQs[:,1,j]
    x_errormin = TQs[:,1,j]-TQs[:,0,j]
  
    x_error = [x_errormin, x_errormax]
  
    figs[j] = plt.figure()
    axs[j] = figs[j].add_subplot(111)
    axs[j].errorbar(x, y,xerr=x_error,fmt='bo') 
    axs[j].plot(x-x_errormin,y,'bx')
    axs[j].plot(x+x_errormax,y,'bx')

    if analysis:
        
        axs[j].errorbar(q50[h], [n_experts+1],xerr=[[q50[h]-q05[h]],[q95[h]-q50[h]]],fmt='ro') 
        axs[j].plot(q05[h],n_experts+1,'rx')
        axs[j].plot(q95[h],n_experts+1,'rx')
 
        axs[j].errorbar([q50_EW[h]], n_experts+2,xerr=[[q50_EW[h]-q05_EW[h]],[q95_EW[h]-q50_EW[h]]],fmt='go') 
        axs[j].plot(q05_EW[h],n_experts+2,'gx')
        axs[j].plot(q95_EW[h],n_experts+2,'gx')

    xt = plt.xticks()[0]  
    xmin, xmax = min(xt), max(xt) 

    if analysis:

        b = np.amin([np.amin(TQs[:,0,j]),q05[h]])
        c = np.amin([np.amax(TQs[:,0,j]),q95[h]])

    else:

        b = np.amin(TQs[:,0,j])
        c = np.amax(TQs[:,0,j])


             
    ytick = []             
    for i in y:
        ytick.append('Exp.'+str(int(i)))
                 
    ytick.append('DM-Cooke')
    
    ytick.append('DM-Equal')
             
    y = np.arange(n_experts+2)+1
                 
    ytick_tuple = tuple(i for i in ytick)
    axs[j].set_yticks(y)
    axs[j].set_yticklabels(ytick_tuple)
    axs[j].set_xlabel(tg_unitok[j])
    plt.title('Target Question '+str(j+1))


    if (np.abs(c-b) >= 9.99e2):
    
        axs[j].set_xscale('log')    
    
    if analysis:
        axs[j].set_ylim(0.5,n_experts+2.5)
    else:
        axs[j].set_ylim(0.5,n_experts+0.5)
    
    
    axs[j].grid(linewidth=0.4)

    figname = path+'/target_'+str(j+1).zfill(2)+'.pdf'
    figs[j].savefig(figname)
    
    figname = path+'/target_'+str(j+1).zfill(2)+'.png'
    figs[j].savefig(figname,dpi=200)

    blank_slide_layout = prs.slide_layouts[6]
    slide = prs.slides.add_slide(title_slide_layout)
    
    
    title_shape = slide.shapes.title
    title_shape.text = TQ_question[j]
    title_para = slide.shapes.title.text_frame.paragraphs[0]
    title_para.font.name = "Helvetica"
    
    img=slide.shapes.add_picture('./'+figname,left,top,width=Inches(10))
    
    shape = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, Inches(0.2),Inches(16),Inches(0.3))
    shape.shadow.inherit = False
    fill=shape.fill
    fill.solid()
    fill.fore_color.rgb=RGBColor(0,0,255)
    shape.text= "Expert elicitation " + datetime.datetime.today().strftime ('%d-%b-%Y')
    shape_para = shape.text_frame.paragraphs[0]
    shape_para.font.name = "Helvetica"
    plt.close()
    
    
    h = h+1


prs.save(path+"/elicitation_old.pptx") # saving file


"""

old_prs = Presentation("elicitation_old.pptx")
new_prs = Presentation()
new_prs.slide_width = Inches(16)
new_prs.slide_height = Inches(9)

copy_slide_from_external_prs(old_prs, 0, new_prs)
copy_slide_from_external_prs(old_prs, 1, new_prs)

k= n_TQ+3
for i in range(n_TQ+n_seed):
    copy_slide_from_external_prs(old_prs, k, new_prs)
    k +=1

copy_slide_from_external_prs(old_prs, n_TQ+2, new_prs)

k = 2
for i in range(n_TQ):
    copy_slide_from_external_prs(old_prs, k, new_prs)
    k +=1
new_prs.save("elicitation.pptx")
#os.remove("elicitation_old.pptx")            
# plt.show()

"""

