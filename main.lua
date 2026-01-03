
local playerWidth,playerHeight 
local player 
player = {}
player.x = 400
player.y = 200
player.targetX = 400
player.targetY = 300
player.speed = 100
player.radius = 20
player.scale = 0.5
player.sprite = love.graphics.newImage('assets/parrot.png')

function love.load()

    -- player.spriteSheet = love.graphics.newImage('assets/player-sheet.png')
    background = love.graphics.newImage('assets/background.png')
end

function love.update(dt)
    --- Calculate the shortest line distance between current positon and target 
    local dx = player.targetX - player.x 
    local dy = player.targetY - player.y 
    local distance  = math.sqrt(dx^2 + dy^2)

    --- move if we are not "close" 
    if distance > 1 then 
        -- normalize and move at a constant speed 
        player.x = player.x  + (dx/distance) * player.speed * dt 
        player.y = player.y  + (dy/distance) * player.speed * dt 
    end 

end

function love.mousepressed(x,y,button,istouch,presses)
    --- if left mouse button is pressed , set new target 

    if button == 1 then 
        player.targetX = x 
        player.targetY = y 
    end 
end

function love.draw()
    -- draw player 
    love.graphics.circle("fill", player.x,player.y, player.radius)
    love.graphics.draw(background, 0 ,0)
    love.graphics.draw(player.sprite,player.x,player.y,0,player.scale,player.scale)
    -- display the info on the screen 
    local screenW,screenH = love.graphics.getDimensions()
    love.graphics.print("Screen Size: ".. screenH .." ".. screenW,10,10)
    -- draw a small dot at the target destination 
    love.graphics.setColor(1,1,1)
    love.graphics.circle('line',player.targetX,player.targetY,5)
end 