import random
import string
def genABC():
    name1="DaniyalAbdulBari"
    name2="TehreemAhmedSiddique"
    seq1=''
    seq2=''
    n=random.randint(30,100)
    #print(f'Lengths s1 {len(seq1)} , s2 {len(seq2)}')1
    with open('a.txt','w') as a,open('b.txt','w') as b,open('c.txt','w') as c:
        for i in range(n):
            a.write(random.choice(name1.upper()))
            b.write(random.choice(name1.upper()))
            c.write(random.choice(name1.upper()))
        a.write('\n')
        b.write('\n')
        c.write('\n')
        for i in range(n):
            a.write(random.choice(name2.upper()))
            b.write(random.choice(name2.upper()))
            c.write(random.choice(name2.upper()))
        a.close()
        b.close()
        c.close()
        # seq2+=random.choice(name2.upper())
        # a.write(seq1)
        # a.write('\n')
        # a.write(seq2)
        # a.close()
        # b.write(seq1)
        # b.write('\n')
        # b.write(seq2)
        # b.close()
        # c.write(seq1)
        # c.write('\n')
        # c.write(seq2)
        # c.close()
def genDEGI():
    n=random.randint(30,100)
    with open('d.txt','w') as d,open('e.txt','w') as e,open('g.txt','w') as g,open('i.txt','w') as i:
        for k in range(n):
            d.write('%d ' %(random.randint(0,100)))
            e.write('%d ' %(random.randint(0,100)))
            g.write('%d ' %(random.randint(0,100)))
            i.write('%d ' %(random.randint(0,100)))
        i.write('\n')
        #18K-1055 and 18K-1064
        Change=[55,64]
        i.write('%d ' %(random.choice(Change)))
        d.close()
        e.close()
        g.close()
        i.close()
def genFH():
    n=random.randint(10,100)
    f=open('f.txt','w')
    h=open('h.txt','w')
    for j in range(2):
        for i in range(n):
            f.write('%d ' %(random.randint(1,100)))
            h.write('%d ' %(random.randint(1,100)))
        f.write('\n')
        h.write('\n')
    Weight=[55,64]
    f.write('%d ' %(random.choice(Weight)))
    h.write('%d ' %(random.choice(Weight)))
    f.close()
    h.close()
def genJ():
    names=["DaniyalAbdulBari","TehreemAhmedSiddique"]
    n=random.randint(5,20)
    j = open('j.txt','w')
    for i in range(n):
        x=random.randint(1,10)
        y=''
        for k in range(x):
            y+=random.choice(string.ascii_lowercase)
        j.write(f'{y} ')
    j.write('\n')
    j.write(names[random.randint(0,1)].lower())
    j.close()

if __name__ == '__main__':
    genABC()
    genDEGI()
    genFH()
    genJ()