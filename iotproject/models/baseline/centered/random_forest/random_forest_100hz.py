def add_vectors(v1, v2):
    return [sum(i) for i in zip(v1, v2)]
def mul_vector_number(v1, num):
    return [i * num for i in v1]
def score(input):
    if input[214] <= -50.04961967468262:
        var0 = [0.0, 0.0, 0.0, 1.0]
    else:
        if input[276] <= 1.616085946559906:
            if input[325] <= -2.1116859316825867:
                var0 = [0.0, 0.0, 1.0, 0.0]
            else:
                if input[253] <= 4.223371982574463:
                    var0 = [1.0, 0.0, 0.0, 0.0]
                else:
                    var0 = [0.0, 0.0, 1.0, 0.0]
        else:
            if input[571] <= 0.08020575996488333:
                var0 = [0.0, 1.0, 0.0, 0.0]
            else:
                var0 = [0.0, 0.0, 1.0, 0.0]
    if input[333] <= -59.732821464538574:
        var1 = [0.0, 0.0, 0.0, 1.0]
    else:
        if input[229] <= 2.970007538795471:
            if input[210] <= 2.3463175296783447:
                var1 = [1.0, 0.0, 0.0, 0.0]
            else:
                if input[495] <= -15.19465434551239:
                    var1 = [0.0, 0.0, 0.0, 1.0]
                else:
                    var1 = [0.0, 1.0, 0.0, 0.0]
        else:
            if input[103] <= 0.07302313484251499:
                var1 = [0.0, 1.0, 0.0, 0.0]
            else:
                var1 = [0.0, 0.0, 1.0, 0.0]
    if input[247] <= 4.3191399574279785:
        if input[216] <= 1.8267759084701538:
            var2 = [1.0, 0.0, 0.0, 0.0]
        else:
            if input[125] <= 26.270992517471313:
                var2 = [0.0, 1.0, 0.0, 0.0]
            else:
                var2 = [0.0, 0.0, 0.0, 1.0]
    else:
        var2 = [0.0, 0.0, 1.0, 0.0]
    if input[235] <= 3.8917750120162964:
        if input[226] <= -49.1984748840332:
            var3 = [0.0, 0.0, 0.0, 1.0]
        else:
            if input[210] <= 2.1368249654769897:
                if input[445] <= -6.106411337852478:
                    var3 = [0.0, 0.0, 1.0, 0.0]
                else:
                    var3 = [1.0, 0.0, 0.0, 0.0]
            else:
                var3 = [0.0, 1.0, 0.0, 0.0]
    else:
        if input[237] <= -1.2824426293373108:
            var3 = [0.0, 1.0, 0.0, 0.0]
        else:
            var3 = [0.0, 0.0, 1.0, 0.0]
    return mul_vector_number(add_vectors(add_vectors(add_vectors(var0, var1), var2), var3), 0.25)
