library(shiny)
library(data.table)
library(ggplot2)
library(curl)
install.packages("vscDebugger")


# read data
gerdracor <- fread("C:\\Users\\richa\\OneDrive\\Studium\\Digital Humanities\\Master\\1. Semester\\Verfahren und Anwendungen in den DH\\Projektarbeit\\Dracor\\dh_group_dracor\\data_files\\corpus_metadata.csv")
gerdracor


# Define UI for application
ui <- fluidPage(

    # Application title
    titlePanel(title = "DraCor App", windowTitle = "DraCor App - DH Project"),

    # Sidebar with input selections for user
    sidebarLayout(
        sidebarPanel(
          selectInput("corpora", "Choose corpus:", list("German", "Italian"), multiple = TRUE, selected = "German"),
        
          selectInput("methods", "Choose a text property:", list("number of speakers", "sentence length", "number of acts", "act length"), multiple = FALSE)
          ,
          sliderInput("years", "Select time frame:", min = 1600, max = 2000, value = c(1600, 2000), sep = "", dragRange = TRUE)
          ),
        
            
        # visulization in main panel
        mainPanel(

          plotOutput("testPlot"),
          plotOutput("testPlot2"),
          #output$value <- renderText({ sliderInput$years })
          textOutput("test")
          
        
    )
))





# Define server logic for application

server <- function(input, output) {
  
  output$testPlot <- renderPlot({
    
    ggplot(gerdracor[], aes(x = yearNormalized, y = numOfSpeakers)) + geom_point()
    
    
  })
  
  output$testPlot2 <- renderPlot({
    
    # generate bins based on input$bins from ui.R
    #x    <- faithful[, 2]
    #bins <- seq(min(x), max(x), length.out = input$bins + 1)
    
    # draw the histogram with the specified number of bins
    ggplot(data=gerdracor[], aes(x = yearNormalized, y = numOfSpeakersMale)) + geom_point()   
    
  })
  
  output$test <- renderText({
    input$years[1]
    
  })
  
  
  
}

# Run the application 
shinyApp(ui = ui, server = server)
