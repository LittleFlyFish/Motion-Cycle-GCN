A=[8.516077041625977,15.025890350341797,29.86135482788086,35.10166931152344,8.06394100189209,17.223495483398438,38.70781326293945,47.39719772338867,7.432133674621582,13.780241012573242,23.61737823486328,28.299480438232422,9.470096588134766,23.234649658203125,44.363197326660156,51.75852966308594,11.842462539672852,24.128551483154297,56.368553161621094,69.4997329711914,13.572973251342773,27.67609405517578,71.5846176147461,90.84364318847656,11.927597999572754,20.785682678222656,37.74137878417969,43.244873046875,7.846477508544922,21.447595596313477,64.77017211914062,81.89273834228516,18.824676513671875,38.4932975769043,66.54153442382812,77.03559112548828,10.311508178710938,24.027395248413086,51.4794807434082,63.60310363769531,10.903478622436523,24.765270233154297,51.639556884765625,62.99449157714844,6.588695526123047,14.875042915344238,38.25090026855469,49.600345611572266,9.580827713012695,22.898574829101562,56.50940704345703,72.45089721679688,25.157363891601562,57.2237548828125,111.9250259399414,132.99412536621094,9.352239608764648,19.798213958740234,38.26649475097656,48.01496124267578]

num = len(A)
a1 = 0
a2 = 0
a3 = 0
a4 = 0
for i in range(num):
    if i%4 == 0:
        a1 = a1+A[i]
    elif i%4 == 1:
        a2 = a2+A[i]
    elif i%4 == 2:
        a3 = a3+A[i]
    else:
        a4 = a4+A[i]

print(a1/15, a2/15, a3/15, a4/15)