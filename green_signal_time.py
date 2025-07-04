def adjust_signal_time(vehicle_count):
    base_green_time=10
    vehicle_multiplier=2
    green_time=base_green_time+(vehicle_count*vehicle_multiplier)
    return green_time

def main():
    try:
        with open("vehicle_count.txt","r")as file:
            vehicle_count=int(file.read())
            if vehicle_count<0:
                print("Invalid vehicle count")
                return
            green_time=adjust_signal_time(vehicle_count)
            print("Adjusted Green Signal Time: ",green_time,"Seconds")
    except(ValueError, FileNotFoundError):
       
        print("Error reading vehicle count from file")

if __name__=="__main__":
    main()
