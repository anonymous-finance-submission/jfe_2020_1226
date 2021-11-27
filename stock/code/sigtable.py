import numpy as np

def sigtable(tablein,coefname,stderrname,pvalname) :
    lines = tablein.split('\\\\\n')
    clines = [ln for ln in lines if coefname in ln]
    slines = [ln for ln in lines if stderrname in ln]
    plines = [ln for ln in lines if pvalname in ln]
    
    newClines = []
    for ln1, ln2  in zip(plines,clines) :
        model = ln2.split('&')[0]
        strng = model + ' & '
        pvals = ln1.split('&')[2:]
        coefs = ln2.split('&')[2:]
        for p, c in zip(pvals,coefs) :
            if float(p)<=0.01 :
                strng += '$'+repr(np.round(float(c),2))+'^{***}$ & '
            elif float(p)<=0.05 :
                strng += '$'+repr(np.round(float(c),2))+'^{**}$ & '
            elif float(p)<=0.1 :
                strng += '$'+repr(np.round(float(c),2))+'^{*}$ & '
            else :
                strng += '$'+repr(np.round(float(c),2))+'$ & '
        strng = '&'.join(strng.split('&')[:-1])
        strng += '\\\\\n'
        newClines.append(strng)
    
    newSlines = []
    for ln in slines :
        serrs = ln.split('&')[2:]
        strng = ' & '
        for s in serrs :
            strng += '$('+repr(np.round(float(s),2))+')$ & '
        strng = '&'.join(strng.split('&')[:-1])
        strng += '\\\\\n'
        newSlines.append(strng)
    
    ln0 = lines[0].replace('&','',1)
    ln0 = ln0.replace('lr','r',1)
    table = ln0 + '\\\\\n' 
    for ln in lines[1:] :
        if (pvalname in ln) or (stderrname in ln): 
            pass
        elif coefname in ln :
            table += newClines.pop(0)
            table += newSlines.pop(0)
        else :
            ln = ln.replace('&','',1)
            table += ln + '\\\\\n' 
    return table