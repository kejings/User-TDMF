
def tda_get(patten_user,train_user_list,test_user_list):
    tda_train_user,tda_train_item = [],[]
    tda_test_user ,tda_test_item= [],[]
    for u in range(len(train_user_list)):
        u_id=train_user_list[u]
        tda_train_user.append(patten_user[u_id])


    for i in range(len(test_user_list)):
        i_id=test_user_list[i]
        tda_test_user.append(patten_user[i_id])


    return tda_train_user,tda_test_user








