from time import time
from gpiozero.pins.pigpio import PiGPIOFactory
from gpiozero import LED
from time import sleep


factory = PiGPIOFactory(host='192.168.106.37')
    
    #Traffic Light 1
green1 = LED(21,pin_factory=factory)
amber1 = LED(20,pin_factory=factory)
red1 = LED(16,pin_factory=factory)

    #Traffic Light 2
green2 = LED(26,pin_factory=factory)
amber2 = LED(13,pin_factory=factory)
red2 = LED(6,pin_factory=factory)

n=0
n1=0
def cars(n):
    cars.var = n
    
def cars1(n1):
    cars1.var = n1
#normal operation
    if cars.var == cars1.var:
        green1.off()
        amber1.off()
        red1.off()
        green2.off()
        amber2.off()
        red2.off()
        for x in range(1):
            sleep(1)
            green1.on()
            red2.on()
            sleep(5)
            green1.off()
            red2.off()
            amber1.on()
            amber2.on()
            sleep(2)
            amber1.off()
            amber2.off()            
            green2.on()
            red1.on()
            sleep(5)
            green2.off()
            red1.off()            
            amber1.on()
            amber2.on()
            sleep(1)
#if lane 1 has zero cars
    elif cars.var == 0:
        green1.off()
        amber1.off()
        red1.on()
        green2.on()
        amber2.off()
        red2.off()
        sleep(2)
#if lane 2 has zero cars   
    elif cars1.var == 0:
        green1.on()
        amber1.off()
        red1.off()
        green2.off()
        amber2.off()
        red2.on()

#if lane 1 has more cars than lane 2            
    elif cars.var > cars1.var:
        green1.off()
        amber1.off()
        red1.off()
        green2.off()
        amber2.off()
        red2.off()
        for x in range(1):
            sleep(1)
            green1.on()
            red2.on()
            sleep(15)
            green1.off()
            red2.off()
            amber1.on()
            amber2.on()
            sleep(2)
            amber1.off()
            amber2.off()
            green2.on()
            red1.on()            
            sleep(5)
            green2.off()
            red1.off()            
            amber1.on()
            amber2.on()
            sleep(1)                       
#if lane 2 has more cars than lane 1
    elif cars.var < cars1.var:
        green1.off()
        amber1.off()
        red1.off()
        green2.off()
        amber2.off()
        red2.off()
        for x in range(1):
            sleep(1)
            green1.on()
            red2.on()
            sleep(5)
            green1.off()
            red2.off()
            amber1.on()
            amber2.on()
            sleep(2)
            amber1.off()
            amber2.off()            
            green2.on()
            red1.on()
            sleep(15)
            amber1.on()
            amber2.on()
            green2.off()
            red1.off()
            sleep(1)

    else:
        green1.off()
        amber1.off()
        red1.off()
        green2.off()
        amber2.off()
        red2.off()
        for x in range(1):
            sleep(1)
            green1.on()
            red2.on()
            sleep(5)
            green1.off()
            red2.off()
            amber1.on()
            amber2.on()
            sleep(2)
            amber1.off()
            amber2.off()            
            green2.on()
            red1.on()
            sleep(5)
            green2.off()
            red1.off()
            amber1.on()
            amber2.on()            
            sleep(1)      
cars(n)
cars1(n1)       