A=[8.732221603393555,16.273555755615234,28.475555419921875,32.334449768066406,8.668506622314453,17.687864303588867,37.483150482177734,44.37425231933594,7.430503845214844,14.771581649780273,26.162227630615234,29.776586532592773,9.282556533813477,21.571331024169922,38.275108337402344,43.872711181640625,11.895395278930664,23.12886619567871,49.320037841796875,59.793190002441406,14.538498878479004,28.478778839111328,66.07025146484375,84.32167053222656,11.717121124267578,19.599220275878906,36.955284118652344,42.3881950378418,9.793777465820312,24.382781982421875,65.77352142333984,82.75989532470703,19.199172973632812,38.66695785522461,62.599159240722656,72.08787536621094,10.287458419799805,24.33552360534668,50.90996551513672,62.521751403808594,10.828038215637207,25.523408889770508,54.63603973388672,66.17550659179688,6.975650310516357,15.38066577911377,41.22651290893555,53.677581787109375,8.862031936645508,22.91912078857422,59.34870147705078,75.17842102050781,31.545242309570312,57.915321350097656,99.155029296875,116.79025268554688,8.89037799835205,18.918027877807617,35.691650390625,45.86875915527344]

num = len(A)
a1 = 0
a2 = 0
a3 = 0
a4 = 0
for i in range(4*15):
    if i%4 == 0:
        a1 = a1+A[i]
    elif i%4 == 1:
        a2 = a2+A[i]
    elif i%4 == 2:
        a3 = a3+A[i]
    else:
        a4 = a4+A[i]

print(a1/15, a2/15, a3/15, a4/15)

#12.696276219679998 25.905436642973328 52.03240496318001 62.61767082213999
