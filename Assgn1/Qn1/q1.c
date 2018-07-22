#include<stdio.h>
int main() {
  int a0 = 1,a1 = 1,a2;
  printf("1: %d\n2: %d\n",a0,a1);
  for (int i = 3;i<=10;i++) {
    a2=a0+a1;
    a0=a1;
    a1=a2;
    printf("%d: %d\n",i,a2);
  }
  return 0;
}
